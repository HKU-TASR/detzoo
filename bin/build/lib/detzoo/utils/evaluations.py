import torch
from torchmetrics import Precision, Recall, F1Score, ROC, AUROC

def precision(gt_labels, pred_scores):
    """
    Compute the precision.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    precision: The precision.
    """
    metric = Precision()
    precision = metric(pred_scores, gt_labels)
    return precision

def recall(gt_labels, pred_scores):
    """
    Compute the recall.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    recall: The recall.
    """
    metric = Recall()
    recall = metric(pred_scores, gt_labels)
    return recall

def f1(gt_labels, pred_scores):
    """
    Compute the F1 score.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    f1: The F1 score.
    """
    metric = F1Score()
    f1 = metric(pred_scores, gt_labels)
    return f1

def roc(gt_labels, pred_scores):
    """
    Compute the ROC curve.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    fpr: False positive rates.
    tpr: True positive rates.
    """
    metric = ROC()
    fpr, tpr, _ = metric(pred_scores, gt_labels)
    return fpr, tpr

def auc(gt_labels, pred_scores):
    """
    Compute the AUC.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    auc: The AUC.
    """
    metric = AUROC()
    auc = metric(pred_scores, gt_labels)
    return auc

def class_AP(gt_labels, pred_scores):
    """
    Compute the average precision for a single class.

    Args:
    gt_labels: Ground truth labels, shape (N, )
    pred_scores: Predicted scores, shape (N, )

    Returns:
    AP: The average precision of the class.
    """
    metric = AveragePrecision(pos_label=1)
    AP = metric(pred_scores, gt_labels)
    return AP

def mAP(gt_labels, pred_scores, num_classes):
    """
    Compute the mean average precision (mAP) over all classes.

    Args:
    gt_labels: Ground truth labels, shape (N, num_classes)
    pred_scores: Predicted scores, shape (N, num_classes)

    Returns:
    mAP: The mean average precision.
    """
    APs = []
    for i in range(num_classes):
        AP = class_AP(gt_labels[:, i], pred_scores[:, i])
        APs.append(AP)
    mAP = torch.mean(torch.stack(APs))
    return mAP

