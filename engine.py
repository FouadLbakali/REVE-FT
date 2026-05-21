import numpy as np
import torch
from tqdm.auto import tqdm

from multilora import set_subject_ids
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


_USE_BF16 = False


def set_bf16(enabled: bool):
    global _USE_BF16
    _USE_BF16 = enabled


def _autocast():
    dtype = torch.bfloat16 if _USE_BF16 else torch.float32
    return torch.amp.autocast(
        dtype=dtype,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )


def _compute_metrics(y_targets, y_decisions, y_probs, n_classes=None, with_ranking=True):
    gt = torch.cat(y_targets).cpu().numpy()
    pr = torch.cat(y_decisions).cpu().numpy()
    pr_probs = torch.softmax(torch.cat(y_probs).float(), dim=1).cpu().numpy()
    if n_classes is None:
        n_classes = pr_probs.shape[1]
    acc = (gt == pr).mean()
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    f1 = f1_score(gt, pr, average="weighted")
    if not with_ranking:
        # Used for tiny per-subject test slices where a class may be absent,
        # which makes AUROC / AUC-PR ill-defined.
        auroc = None
        auc_pr = None
    elif n_classes == 2:
        auroc = roc_auc_score(gt, pr_probs[:, 1])
        auc_pr = average_precision_score(gt, pr_probs[:, 1])
    else:
        auroc = roc_auc_score(gt, pr_probs, multi_class='ovr', labels=list(range(n_classes)))
        auc_pr = average_precision_score(label_binarize(gt, classes=range(n_classes)), pr_probs, average='macro')
    return {"acc": acc, "balanced_acc": balanced_acc, "cohen_kappa": cohen_kappa,
            "f1": f1, "auroc": auroc, "auc_pr": auc_pr}


def _metrics_by_subject(gt, dec, prob, subj, n_classes=None):
    """Group cached predictions by subject id and compute per-subject metrics.

    gt / dec / prob / subj are 1-D / 2-D CPU tensors aligned on the sample axis.
    Returns {subject_id (as stored, 0-based): metrics dict incl. n_test}.
    """
    out = {}
    for s in torch.unique(subj).tolist():
        m = subj == s
        try:
            metrics = _compute_metrics([gt[m]], [dec[m]], [prob[m]], n_classes)
        except ValueError:
            metrics = _compute_metrics([gt[m]], [dec[m]], [prob[m]], n_classes,
                                       with_ranking=False)
        # Tiny per-subject slices can yield ill-defined metrics (e.g. an absent
        # class -> NaN AUROC); store None so the dumped JSON stays valid.
        metrics = {k: (None if v is None or not np.isfinite(v) else v)
                   for k, v in metrics.items()}
        metrics["n_test"] = int(m.sum())
        out[int(s)] = metrics
    return out


def train_one_epoch(model, optimizer, loader, device, use_subject_id=False, scheduler=None):
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
        with _autocast():
            if use_subject_id:
                subject_id = batch_data["subject_id"].to(device, non_blocking=True)
                output = model(data, pos, subject_id)
            else:
                output = model(data, pos)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * target.size(0)
        correct += (output.argmax(dim=1) == target).sum().item()
        count += target.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / count, correct / count

def eval_model(model, loader, device, n_classes=None, use_subject_id=False):
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
            with _autocast():
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
def eval_model_per_subject(model, loader, device, n_classes=None, use_subject_id=False):
    """Like eval_model but also returns per-subject metrics.

    Returns (pooled_metrics, {subject_id: metrics}).
    """
    model.eval()

    y_decisions, y_targets, y_probs, y_subjects = [], [], [], []
    pbar = tqdm(loader, desc="Evaluating", total=len(loader))
    for batch_data in pbar:
        data, target, pos = (
            batch_data["sample"].to(device, non_blocking=True),
            batch_data["label"].to(device, non_blocking=True),
            batch_data["pos"].to(device, non_blocking=True),
        )
        with _autocast():
            if use_subject_id:
                subject_id = batch_data["subject_id"].to(device, non_blocking=True)
                output = model(data, pos, subject_id)
            else:
                output = model(data, pos)

        y_decisions.append(torch.argmax(output, dim=1))
        y_targets.append(target)
        y_probs.append(output)
        y_subjects.append(batch_data["subject_id"])

    pooled = _compute_metrics(y_targets, y_decisions, y_probs, n_classes)
    gt = torch.cat(y_targets).cpu()
    dec = torch.cat(y_decisions).cpu()
    prob = torch.cat(y_probs).cpu()
    subj = torch.cat(y_subjects).cpu()
    return pooled, _metrics_by_subject(gt, dec, prob, subj, n_classes)


@torch.inference_mode()
def extract_features(model, loader, device, return_subjects=False):
    """Run a single forward pass over the loader and return cached (features, labels).

    Temporarily swaps model.final_layer with nn.Identity so the forward returns
    the pre-head features of shape (B, C, H, E). Useful for linear probing, where
    the backbone is frozen and features can be reused across epochs.

    If return_subjects is True, also returns the per-sample subject ids.
    """
    model.eval()
    original_final_layer = model.final_layer
    model.final_layer = torch.nn.Identity()
    model.to(device)

    all_features = []
    all_labels = []
    all_subjects = []
    pbar = tqdm(loader, desc="Extracting features", total=len(loader))
    for batch_data in pbar:
        data = batch_data["sample"].to(device, non_blocking=True)
        pos = batch_data["pos"].to(device, non_blocking=True)
        with _autocast():
            feats = model(data, pos)
        all_features.append(feats.float().cpu())
        all_labels.append(batch_data["label"].cpu())
        if return_subjects:
            all_subjects.append(batch_data["subject_id"].cpu())

    model.final_layer = original_final_layer
    if return_subjects:
        return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_subjects)
    return torch.cat(all_features), torch.cat(all_labels)


def train_head_one_epoch(head, optimizer, features, labels, batch_size, device,
                         scheduler=None):
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
        out = head(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / n, correct / n


def eval_head(head, features, labels, device, batch_size=256, n_classes=None):
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
            out = head(x)
            decisions = torch.argmax(out, dim=1)
            y_decisions.append(decisions)
            y_targets.append(y)
            y_probs.append(out)

    return _compute_metrics(y_targets, y_decisions, y_probs, n_classes)


def eval_head_per_subject(head, features, labels, subjects, device,
                          batch_size=256, n_classes=None):
    """Like eval_head but also returns per-subject metrics.

    `subjects` is the per-sample subject id tensor aligned with `features`.
    Returns (pooled_metrics, {subject_id: metrics}).
    """
    head.eval()
    n = features.size(0)
    y_decisions, y_targets, y_probs = [], [], []
    n_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(range(0, n, batch_size), desc="Evaluating", total=n_batches)
    with torch.inference_mode():
        for start in pbar:
            x = features[start:start + batch_size].to(device, non_blocking=True)
            y = labels[start:start + batch_size].to(device, non_blocking=True)
            out = head(x)
            y_decisions.append(torch.argmax(out, dim=1))
            y_targets.append(y)
            y_probs.append(out)

    pooled = _compute_metrics(y_targets, y_decisions, y_probs, n_classes)
    gt = torch.cat(y_targets).cpu()
    dec = torch.cat(y_decisions).cpu()
    prob = torch.cat(y_probs).cpu()
    return pooled, _metrics_by_subject(gt, dec, prob, subjects.cpu(), n_classes)


# --------------------------------------------------------------------------- #
# Multi-subject LoRA: subject-mixed batches, per-sample adapter routing.       #
# The subject ids are pushed into a shared context so the multi-LoRA layers    #
# pick the right adapter per sample within a single forward/backward.          #
# --------------------------------------------------------------------------- #

def train_one_epoch_multilora(model, optimizer, loader, device, scheduler=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    pbar = tqdm(loader, desc="Training", total=len(loader))

    total_loss, correct, count = 0.0, 0, 0
    for batch_data in pbar:
        data = batch_data["sample"].to(device, non_blocking=True)
        target = batch_data["label"].to(device, non_blocking=True)
        pos = batch_data["pos"].to(device, non_blocking=True)
        set_subject_ids(batch_data["subject_id"].to(device, non_blocking=True))

        optimizer.zero_grad()
        with _autocast():
            output = model(data, pos)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * target.size(0)
        correct += (output.argmax(dim=1) == target).sum().item()
        count += target.size(0)
        pbar.set_postfix({"loss": loss.item()})

    set_subject_ids(None)
    return total_loss / count, correct / count


@torch.inference_mode()
def eval_model_multilora(model, loader, device, n_classes=None):
    model.eval()
    y_decisions, y_targets, y_probs = [], [], []
    pbar = tqdm(loader, desc="Evaluating", total=len(loader))
    for batch_data in pbar:
        data = batch_data["sample"].to(device, non_blocking=True)
        target = batch_data["label"].to(device, non_blocking=True)
        pos = batch_data["pos"].to(device, non_blocking=True)
        set_subject_ids(batch_data["subject_id"].to(device, non_blocking=True))
        with _autocast():
            output = model(data, pos)
        y_decisions.append(torch.argmax(output, dim=1))
        y_targets.append(target)
        y_probs.append(output)

    set_subject_ids(None)
    return _compute_metrics(y_targets, y_decisions, y_probs, n_classes)


@torch.inference_mode()
def eval_model_multilora_per_subject(model, loader, device, n_classes=None):
    """Like eval_model_multilora but also returns per-subject metrics."""
    model.eval()
    y_decisions, y_targets, y_probs, y_subjects = [], [], [], []
    pbar = tqdm(loader, desc="Evaluating", total=len(loader))
    for batch_data in pbar:
        data = batch_data["sample"].to(device, non_blocking=True)
        target = batch_data["label"].to(device, non_blocking=True)
        pos = batch_data["pos"].to(device, non_blocking=True)
        set_subject_ids(batch_data["subject_id"].to(device, non_blocking=True))
        with _autocast():
            output = model(data, pos)
        y_decisions.append(torch.argmax(output, dim=1))
        y_targets.append(target)
        y_probs.append(output)
        y_subjects.append(batch_data["subject_id"])

    set_subject_ids(None)
    pooled = _compute_metrics(y_targets, y_decisions, y_probs, n_classes)
    gt = torch.cat(y_targets).cpu()
    dec = torch.cat(y_decisions).cpu()
    prob = torch.cat(y_probs).cpu()
    subj = torch.cat(y_subjects).cpu()
    return pooled, _metrics_by_subject(gt, dec, prob, subj, n_classes)
