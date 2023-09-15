import numpy as np
#import torch
import jittor as jt


def IoU(mask1, mask2):
    mask1, mask2 = mask1>0.5, mask2>0.5
    intersection = jt.sum(mask1 * (mask1 == mask2))
    union = jt.sum(mask1 + mask2)
    return (jt.float32(intersection) / union).mean().item()


def accuracy(mask1, mask2):
    mask1, mask2 = jt.to_bool(mask1>0.5), jt.to_bool(mask2>0.5)
    return jt.mean((mask1 == mask2).to(jt.float32)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = jt.to_bool(mask_gt), jt.to_bool(mask)
    true_positive = jt.sum(mask_gt * (mask_gt == mask), dim=(-1, -2)).squeeze()
    mask_area = jt.sum(mask, dim=(-1, -2))
    mask_gt_area = jt.sum(mask_gt, dim=(-1, -2))

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    return F.mean(dim=0).max().item()

@jt.no_grad()
def metrics(pred, gt, stats=(IoU, accuracy, F_max), prob_bins=255):
    avg_values = {}
    precisions = []
    recalls = []
    out_dict = {}
    
    nb_sample = len(gt)
    for step in range(nb_sample):
        prediction, mask = jt.array(pred[step]), jt.array(gt[step])

        for metric in stats:
            method = metric.__name__
            if method not in avg_values and metric != F_max:
                avg_values[method] = 0.0

            if metric != F_max:
                avg_values[method] += metric(mask, prediction)
            else:
                p, r = [], []
                splits = 2.0 * jt.mean(prediction, dim=0) if prob_bins is None else \
                    np.arange(0.0, 1.0, 1.0 / prob_bins)

                for split in splits:
                    pr = precision_recall(mask, prediction > split)
                    p.append(pr[0])
                    r.append(pr[1])
                precisions.append(p)
                recalls.append(r)

    for metric in stats:
        method = metric.__name__
        if metric == F_max:
            out_dict[method] = F_max(jt.array(precisions), jt.array(recalls))
        else:
            out_dict[method] = avg_values[method] / nb_sample

    return out_dict