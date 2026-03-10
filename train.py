import argparse
from transformers import AutoModel
import torch
from transformers import set_seed
from data import load_bciciv2a
from engine import train_one_epoch, eval_model
import sys, types
sys.path.insert(0, "reve-repro-main/src")
from models.lora import get_lora_config, CustomGetLora
import types

def main():
    parser = argparse.ArgumentParser(description='REVE Training')
    parser.add_argument('--mode', default="linear", choices=["linear", "full", "lora"])
    parser.add_argument('--epochs', default=20, type=int, help='number of epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--seed', default=None, type=int, help='seed')
    args = parser.parse_args()

    # Loading REVE
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    # Add Classification Head
    dim = 22 * 5 * 512
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(dim, 4),
    )

    # Load Dataset
    train_loader, val_loader, test_loader = load_bciciv2a(pos_bank, args.batch_size, seed=args.seed)

    mode = args.mode
    if mode == "linear":
        params = list(model.final_layer.parameters())
    elif mode == "full":
        params = list(model.parameters())
    elif mode == "lora":
        lora_config = get_lora_config(types.SimpleNamespace(encoder=model), rank=8, apply_to=("patch", "mlp4d", "attention", "ffw"))
        model = CustomGetLora(lora_config).get_model(model)
        model.final_layer.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]

    total = sum(p.numel() for p in params)
    print(f"Total paramètres : {total:,}")

    if args.seed:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0
    best_final_layer = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, optimizer, train_loader, device)
        b_acc = eval_model(model, val_loader, device)["balanced_acc"]
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_final_layer = model.final_layer.state_dict()
        print(f"Validation balanced accuracy: {b_acc:.4f}, best: {best_val_acc:.4f}")
        scheduler.step(b_acc)


    model.final_layer.load_state_dict(best_final_layer)
    results = eval_model(model, test_loader, device)

    # Results
    print("acc", results["acc"])
    print("balanced_acc", results["balanced_acc"])
    print("cohen_kappa", results["cohen_kappa"])
    print("f1", results["f1"])
    print("auroc", results["auroc"])
    print("auc_pr", results["auc_pr"])

if __name__ == "__main__":
    main()

