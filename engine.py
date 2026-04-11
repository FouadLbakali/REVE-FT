import torch
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def _compute_metrics(y_targets, y_decisions, y_probs, n_classes=4):
    gt = torch.cat(y_targets).cpu().numpy()
    pr = torch.cat(y_decisions).cpu().numpy()
    pr_probs = torch.softmax(torch.cat(y_probs).float(), dim=1).cpu().numpy()
    acc = (gt == pr).mean()
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    f1 = f1_score(gt, pr, average="weighted")
    auroc = roc_auc_score(gt, pr_probs, multi_class='ovr')
    auc_pr = average_precision_score(label_binarize(gt, classes=range(n_classes)), pr_probs, average='macro')
    return {"acc": acc, "balanced_acc": balanced_acc, "cohen_kappa": cohen_kappa,
            "f1": f1, "auroc": auroc, "auc_pr": auc_pr}


def train_one_epoch(model, optimizer, loader, device, use_subject_id=False, l1_lambda=0.0):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    pbar = tqdm(loader, desc="Training", total=len(loader))

    total_loss, correct, count = 0.0, 0, 0
    for batch_data in pbar:
        data, target, pos = (
            batch_data["sample"].to(device, non_blocking=True),
            batch_data["label"].to(device, non_blocking=True),
            batch_data["pos"].to(device, non_blocking=True),
        )
        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda" if torch.cuda.is_available() else "cpu"):
            if use_subject_id:
                subject_id = batch_data["subject_id"].to(device, non_blocking=True)
                output = model(data, pos, subject_id)
            else:
                output = model(data, pos)
        loss = criterion(output, target)
        if l1_lambda > 0:
            l1 = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l1_lambda * l1
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        correct += (output.argmax(dim=1) == target).sum().item()
        count += target.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / count, correct / count

def eval_model(model, loader, device, n_classes=4, use_subject_id=False):
    model.eval()

    y_decisions = []
    y_targets = []
    y_probs = []
    score, count = 0, 0
    pbar = tqdm(loader, desc="Evaluating", total=len(loader))
    with torch.inference_mode():
        for batch_data in pbar:
            data, target, pos = (
                batch_data["sample"].to(device, non_blocking=True),
                batch_data["label"].to(device, non_blocking=True),
                batch_data["pos"].to(device, non_blocking=True),
            )
            with torch.amp.autocast(
                dtype=torch.float16, device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                if use_subject_id:
                    subject_id = batch_data["subject_id"].to(device, non_blocking=True)
                    output = model(data, pos, subject_id)
                else:
                    output = model(data, pos)

            decisions = torch.argmax(output, dim=1)
            score += (decisions == target).int().sum().item()
            count += target.shape[0]
            y_decisions.append(decisions)
            y_targets.append(target)
            y_probs.append(output)

    return _compute_metrics(y_targets, y_decisions, y_probs, n_classes)


@torch.inference_mode()
def extract_features(model, loader, device):
    """Run a single forward pass over the loader and return cached (features, labels).

    Temporarily swaps model.final_layer with nn.Identity so the forward returns
    the pre-head features of shape (B, C, H, E). Useful for linear probing, where
    the backbone is frozen and features can be reused across epochs.
    """
    model.eval()
    original_final_layer = model.final_layer
    model.final_layer = torch.nn.Identity()
    model.to(device)

    all_features = []
    all_labels = []
    pbar = tqdm(loader, desc="Extracting features", total=len(loader))
    for batch_data in pbar:
        data = batch_data["sample"].to(device, non_blocking=True)
        pos = batch_data["pos"].to(device, non_blocking=True)
        with torch.amp.autocast(
            dtype=torch.float16, device_type="cuda" if torch.cuda.is_available() else "cpu"
        ):
            feats = model(data, pos)
        all_features.append(feats.float().cpu())
        all_labels.append(batch_data["label"].cpu())

    model.final_layer = original_final_layer
    return torch.cat(all_features), torch.cat(all_labels)


def train_head_one_epoch(head, optimizer, features, labels, batch_size, device, l1_lambda=0.0):
    """Train the classification head for one epoch on cached features."""
    criterion = torch.nn.CrossEntropyLoss()
    head.train()
    n = features.size(0)
    perm = torch.randperm(n)
    total_loss, correct = 0.0, 0
    n_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(range(0, n, batch_size), desc="Training", total=n_batches)
    for start in pbar:
        idx = perm[start:start + batch_size]
        x = features[idx].to(device, non_blocking=True)
        y = labels[idx].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(
            dtype=torch.float16, device_type="cuda" if torch.cuda.is_available() else "cpu"
        ):
            out = head(x)
        loss = criterion(out, y)
        if l1_lambda > 0:
            l1 = sum(p.abs().sum() for p in head.parameters() if p.requires_grad)
            loss = loss + l1_lambda * l1
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / n, correct / n


def eval_head(head, features, labels, device, batch_size=256, n_classes=4):
    """Evaluate the classification head on cached features."""
    head.eval()
    n = features.size(0)
    y_decisions, y_targets, y_probs = [], [], []
    n_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(range(0, n, batch_size), desc="Evaluating", total=n_batches)
    with torch.inference_mode():
        for start in pbar:
            x = features[start:start + batch_size].to(device, non_blocking=True)
            y = labels[start:start + batch_size].to(device, non_blocking=True)
            with torch.amp.autocast(
                dtype=torch.float16, device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                out = head(x)
            decisions = torch.argmax(out, dim=1)
            y_decisions.append(decisions)
            y_targets.append(y)
            y_probs.append(out)

    return _compute_metrics(y_targets, y_decisions, y_probs, n_classes)
