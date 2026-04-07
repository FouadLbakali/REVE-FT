import torch
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

def train_one_epoch(model, optimizer, loader, device, use_subject_id=False):
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

    gt = torch.cat(y_targets).cpu().numpy()
    pr = torch.cat(y_decisions).cpu().numpy()
    pr_probs = torch.softmax(torch.cat(y_probs).float(), dim=1).cpu().numpy()
    acc = score / count
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    f1 = f1_score(gt, pr, average="weighted")

    auroc = roc_auc_score(gt, pr_probs, multi_class='ovr')
    auc_pr = average_precision_score(label_binarize(gt, classes=range(n_classes)), pr_probs, average='macro')

    return {"acc": acc, "balanced_acc": balanced_acc, "cohen_kappa": cohen_kappa,
            "f1": f1, "auroc": auroc, "auc_pr": auc_pr}
