import torch
import numpy as np
from tqdm import tqdm
from config import Config


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            target_boxes = batch['boxes']
            target_labels = batch['labels']

            # Get model predictions
            outputs = model(images)
            pred_boxes, pred_labels = process_predictions(outputs)

            all_predictions.extend(zip(pred_boxes, pred_labels))
            all_targets.extend(zip(target_boxes, target_labels))

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    return metrics


def process_predictions(outputs):
    """Process raw model outputs into boxes and labels"""
    if isinstance(outputs, tuple):  # SSD output
        classifications, boxes = outputs
        pred_boxes = boxes.cpu().numpy()
        pred_labels = classifications.argmax(dim=1).cpu().numpy()
    else:  # YOLO output
        # Process YOLO outputs
        pred_boxes = []
        pred_labels = []
        for output in outputs:
            # Process each detection scale
            boxes, labels = process_yolo_output(output)
            pred_boxes.extend(boxes)
            pred_labels.extend(labels)

    return pred_boxes, pred_labels


def calculate_metrics(predictions, targets):
    """Calculate precision, recall, mAP, and IoU"""
    ious = []
    correct_detections = 0
    total_predictions = len(predictions)
    total_targets = len(targets)

    class_metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in Config.CLASSES}

    for pred, target in zip(predictions, targets):
        pred_box, pred_label = pred
        target_box, target_label = target

        iou = calculate_iou(pred_box, target_box)
        ious.append(iou)

        if iou > Config.CONFIDENCE_THRESHOLD:
            correct_detections += 1
            if pred_label == target_label:
                class_metrics[Config.CLASSES[pred_label]]['TP'] += 1
            else:
                class_metrics[Config.CLASSES[pred_label]]['FP'] += 1
                class_metrics[Config.CLASSES[target_label]]['FN'] += 1
        else:
            class_metrics[Config.CLASSES[pred_label]]['FP'] += 1
            class_metrics[Config.CLASSES[target_label]]['FN'] += 1

    # Calculate overall metrics
    precision = correct_detections / total_predictions if total_predictions > 0 else 0
    recall = correct_detections / total_targets if total_targets > 0 else 0
    mean_iou = np.mean(ious)

    # Calculate mAP
    ap_per_class = []
    for cls in Config.CLASSES:
        metrics = class_metrics[cls]
        cls_precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
        cls_recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
        ap_per_class.append((cls_precision + cls_recall) / 2)  # Simplified AP calculation

    mAP = np.mean(ap_per_class)

    return {
        'precision': precision,
        'recall': recall,
        'mAP': mAP,
        'IoU': mean_iou,
        'per_class_metrics': class_metrics
    }