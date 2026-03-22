import argparse
import sys
import types

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel, set_seed

from data import load_bciciv2a
from engine import train_one_epoch, eval_model

sys.path.insert(0, "reve-repro-main/src")
from models.lora import get_lora_config, CustomGetLora


# ──────────────────────────────────────────────────────────────────────────────
# Subject-Specific Prompt Model
# ──────────────────────────────────────────────────────────────────────────────

class ReveSubjectPrompt(nn.Module):
    def __init__(self, reve_model, num_subject, num_prompt=1, num_classes=4, dropout=0.1):
        super().__init__()
        self.reve = reve_model
        self.embed_dim = reve_model.embed_dim
        self.num_subject = num_subject
        self.num_prompt = num_prompt

        for param in self.reve.parameters():
            param.requires_grad = False

        self.subject_prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_prompt, self.embed_dim))
            for _ in range(num_subject)
        ])

        self.classifier = nn.Sequential(
            nn.RMSNorm(num_prompt * self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(num_prompt * self.embed_dim, num_classes),
        )

    def freeze_prompts(self):
        for p in self.subject_prompt_tokens:
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def freeze_classifier(self):
        for p in self.subject_prompt_tokens:
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.subject_prompt_tokens:
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def subject_attention_pooling(self, x, subject_ids):
        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")

        queries = torch.stack([
            self.subject_prompt_tokens[sid.item()] for sid in subject_ids
        ]).squeeze(1)

        attention_scores = torch.matmul(queries, x.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, x)

        return out.reshape(b, -1)

    def forward(self, eeg, pos, subject_ids):
        with torch.no_grad():
            x = self.reve(eeg, pos)
        pooled = self.subject_attention_pooling(x, subject_ids)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Shared Query Token Model (baseline attention pooling)
# ──────────────────────────────────────────────────────────────────────────────

class ReveQueryToken(nn.Module):
    def __init__(self, reve_model, num_prompt=1, num_classes=4, dropout=0.1):
        super().__init__()
        self.reve = reve_model
        self.embed_dim = reve_model.embed_dim
        self.num_prompt = num_prompt

        for param in self.reve.parameters():
            param.requires_grad = False

        self.query_tokens = nn.Parameter(torch.randn(1, num_prompt, self.embed_dim))

        self.classifier = nn.Sequential(
            nn.RMSNorm(num_prompt * self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(num_prompt * self.embed_dim, num_classes),
        )

    def attention_pooling(self, x):
        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")
        queries = self.query_tokens.expand(b, -1, -1)
        attention_scores = torch.matmul(queries, x.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, x)
        return out.reshape(b, -1)

    def forward(self, eeg, pos):
        with torch.no_grad():
            x = self.reve(eeg, pos)
        pooled = self.attention_pooling(x)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_training_loop(model, train_loader, val_loader, device, params, lr, n_epochs,
                      use_subject_id=False, stage_name="", loss_history=None):
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    if loss_history is None:
        loss_history = []

    best_val_acc = 0
    best_state = None

    for epoch in range(n_epochs):
        prefix = f"[{stage_name}] " if stage_name else ""
        print(f"{prefix}Epoch {epoch + 1}/{n_epochs}")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, use_subject_id=use_subject_id)
        loss_history.append(avg_loss)
        b_acc = eval_model(model, val_loader, device, use_subject_id=use_subject_id)["balanced_acc"]
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"{prefix}Val balanced acc: {b_acc:.4f}, best: {best_val_acc:.4f}  loss: {avg_loss:.4f}")
        scheduler.step(b_acc)

    model.load_state_dict(best_state)
    return best_val_acc, loss_history


def save_loss_curve(loss_history, save_path, title="Training Loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")

    csv_path = save_path.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(loss_history, 1):
            f.write(f"{i},{loss}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='REVE Training v2')
    parser.add_argument('--mode', default="linear",
                        choices=["linear", "full", "lora", "query_token", "subject_prompt"])
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs (per stage for twostage)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--seed', default=None, type=int, help='seed')
    parser.add_argument('--num-prompt', default=4, type=int, help='number of prompt tokens per subject')
    parser.add_argument('--num-subject', default=9, type=int, help='number of subjects')
    args = parser.parse_args()

    # Seed
    if args.seed:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Load REVE
    reve_model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    # Load data
    train_loader, val_loader, test_loader = load_bciciv2a(pos_bank, args.batch_size, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode
    use_subject_id = mode == "subject_prompt"

    # ── Build model depending on mode ──────────────────────────────────────
    if mode in ("linear", "full", "lora"):
        model = reve_model
        dim = 22 * 5 * 512
        model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, 4),
        )

        if mode == "linear":
            params = list(model.final_layer.parameters())
        elif mode == "full":
            params = list(model.parameters())
        elif mode == "lora":
            lora_config = get_lora_config(
                types.SimpleNamespace(encoder=model), rank=8,
                apply_to=("patch", "mlp4d", "attention", "ffw"),
            )
            model = CustomGetLora(lora_config).get_model(model)
            model.final_layer.requires_grad_(True)
            params = [p for p in model.parameters() if p.requires_grad]

        print(f"Mode: {mode} — trainable params: {sum(p.numel() for p in params):,}")
        model.to(device)
        _, loss_history = run_training_loop(model, train_loader, val_loader, device, params, args.lr, args.epochs)
        title = f"{mode}"

    elif mode == "query_token":
        model = ReveQueryToken(reve_model, num_prompt=args.num_prompt, num_classes=4)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"Mode: query_token (num_prompt={args.num_prompt}) — trainable params: {sum(p.numel() for p in params):,}")
        model.to(device)
        _, loss_history = run_training_loop(model, train_loader, val_loader, device, params, args.lr, args.epochs)
        title = f"query_token  num_prompt={args.num_prompt}"

    elif mode == "subject_prompt":
        model = ReveSubjectPrompt(reve_model, num_subject=args.num_subject,
                                  num_prompt=args.num_prompt, num_classes=4)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"Mode: subject_prompt — trainable params: {sum(p.numel() for p in params):,}")
        model.to(device)
        _, loss_history = run_training_loop(model, train_loader, val_loader, device, params, args.lr, args.epochs,
                                            use_subject_id=True)
        title = f"subject_prompt  num_prompt={args.num_prompt}"

    # ── Loss curve ─────────────────────────────────────────────────────────
    curve_name = f"loss_{mode}"
    if mode in ("query_token", "subject_prompt"):
        curve_name += f"_np{args.num_prompt}"
    save_loss_curve(loss_history, f"{curve_name}.png", title=title)

    # ── Test evaluation ────────────────────────────────────────────────────
    results = eval_model(model, test_loader, device, use_subject_id=use_subject_id)

    print("\n--- Test Results ---")
    print(f"Accuracy:          {results['acc']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_acc']:.4f}")
    print(f"Cohen's Kappa:     {results['cohen_kappa']:.4f}")
    print(f"F1 (weighted):     {results['f1']:.4f}")
    print(f"AUROC:             {results['auroc']:.4f}")
    print(f"AUC-PR:            {results['auc_pr']:.4f}")


if __name__ == "__main__":
    main()
