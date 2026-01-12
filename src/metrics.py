import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred_ids, gold_ids, id2answer):
    pred_ids = np.array(pred_ids)
    gold_ids = np.array(gold_ids)

    overall_acc = float((pred_ids == gold_ids).mean())

    gold_ans = np.array([id2answer[int(i)] for i in gold_ids])
    yesno_mask = np.isin(gold_ans, ["yes", "no"])
    open_mask = ~yesno_mask

    metrics = {"overall_accuracy": overall_acc}

    if yesno_mask.any():
        y_true = gold_ans[yesno_mask]
        y_pred = np.array([id2answer[int(i)] for i in pred_ids[yesno_mask]])
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["yes", "no"], average="macro", zero_division=0
        )
        metrics.update({
            "yesno_accuracy": float(acc),
            "yesno_precision_macro": float(pr),
            "yesno_recall_macro": float(rc),
            "yesno_f1_macro": float(f1),
        })
    else:
        metrics.update({
            "yesno_accuracy": None,
            "yesno_precision_macro": None,
            "yesno_recall_macro": None,
            "yesno_f1_macro": None,
        })

    if open_mask.any():
        open_acc = float((pred_ids[open_mask] == gold_ids[open_mask]).mean())
        metrics["openended_accuracy"] = open_acc
    else:
        metrics["openended_accuracy"] = None

    return metrics
