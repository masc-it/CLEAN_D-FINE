# from supervision, didn't need the extra features / bloat, https://github.com/roboflow/supervision

from dataclasses import dataclass
from typing import List

import numpy as np


def validate_input_tensors(predictions: List[np.ndarray], targets: List[np.ndarray]):
    """
    Checks for shape consistency of input tensors.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) and"
            f"targets ({len(targets)}) must be equal."
        )
    if len(predictions) > 0:
        if not isinstance(predictions[0], np.ndarray) or not isinstance(
            targets[0], np.ndarray
        ):
            raise ValueError(
                f"Predictions and targets must be lists of numpy arrays."
                f"Got {type(predictions[0])} and {type(targets[0])} instead."
            )
        if predictions[0].shape[1] != 6:
            raise ValueError(
                f"Predictions must have shape (N, 6)."
                f"Got {predictions[0].shape} instead."
            )
        if targets[0].shape[1] != 5:
            raise ValueError(
                f"Targets must have shape (N, 5). Got {targets[0].shape} instead."
            )


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    ious = area_inter / (area_true[:, None] + area_detection - area_inter)
    ious = np.nan_to_num(ious)
    return ious


@dataclass(frozen=True)
class MeanAveragePrecision:
    """
    Mean Average Precision for object detection tasks.

    Attributes:
        map50_95 (float): Mean Average Precision (mAP) calculated over IoU thresholds
            ranging from `0.50` to `0.95` with a step size of `0.05`.
        map50 (float): Mean Average Precision (mAP) calculated specifically at
            an IoU threshold of `0.50`.
        map75 (float): Mean Average Precision (mAP) calculated specifically at
            an IoU threshold of `0.75`.
        per_class_ap50_95 (np.ndarray): Average Precision (AP) values calculated over
            IoU thresholds ranging from `0.50` to `0.95` with a step size of `0.05`,
            provided for each individual class.
    """

    map50_95: float
    map50: float
    map75: float
    per_class_ap50_95: np.ndarray
    per_class_p50_95: np.ndarray
    per_class_r50_95: np.ndarray

    @classmethod
    def from_tensors(
        cls,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> "MeanAveragePrecision":
        """
        Calculate Mean Average Precision based on predicted and ground-truth
            detections at different threshold.

        Args:
            predictions (List[np.ndarray]): Each element of the list describes
                a single image and has `shape = (M, 6)` where `M` is
                the number of detected objects. Each row is expected to be
                in `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (List[np.ndarray]): Each element of the list describes a single
                image and has `shape = (N, 5)` where `N` is the
                number of ground-truth objects. Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
        Returns:
            MeanAveragePrecision: New instance of MeanAveragePrecision.

        Example:
            ```python
            import supervision as sv
            import numpy as np

            targets = (
                [
                    np.array(
                        [
                            [0.0, 0.0, 3.0, 3.0, 1],
                            [2.0, 2.0, 5.0, 5.0, 1],
                            [6.0, 1.0, 8.0, 3.0, 2],
                        ]
                    ),
                    np.array([[1.0, 1.0, 2.0, 2.0, 2]]),
                ]
            )

            predictions = [
                np.array(
                    [
                        [0.0, 0.0, 3.0, 3.0, 1, 0.9],
                        [0.1, 0.1, 3.0, 3.0, 0, 0.9],
                        [6.0, 1.0, 8.0, 3.0, 1, 0.8],
                        [1.0, 6.0, 2.0, 7.0, 1, 0.8],
                    ]
                ),
                np.array([[1.0, 1.0, 2.0, 2.0, 2, 0.8]])
            ]

            mean_average_precison = sv.MeanAveragePrecision.from_tensors(
                predictions=predictions,
                targets=targets,
            )

            print(mean_average_precison.map50_95)
            # 0.6649
            ```
        """
        validate_input_tensors(predictions, targets)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        # Gather matching stats for predictions and targets
        for true_objs, predicted_objs in zip(targets, predictions):
            if predicted_objs.shape[0] == 0:
                if true_objs.shape[0]:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            *np.zeros((2, 0)),
                            true_objs[:, 4],
                        )
                    )
                continue

            if true_objs.shape[0]:
                matches = cls._match_detection_batch(
                    predicted_objs, true_objs, iou_thresholds
                )
                stats.append(
                    (
                        matches,
                        predicted_objs[:, 5],
                        predicted_objs[:, 4],
                        true_objs[:, 4],
                    )
                )

        # Compute average precisions if any matches exist
        if stats:
            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions, precisions, recalls = cls._average_precisions_per_class(
                *concatenated_stats
            )  # type: ignore
            map50 = average_precisions[:, 0].mean()
            map75 = average_precisions[:, 5].mean()
            map50_95 = average_precisions.mean()
        else:
            map50, map75, map50_95 = 0, 0, 0
            average_precisions = []
            precisions = []
            recalls = []

        return cls(
            map50_95=map50_95,
            map50=map50,
            map75=map75,
            per_class_ap50_95=average_precisions,  # type: ignore
            per_class_p50_95=precisions,  # type: ignore
            per_class_r50_95=recalls,  # type: ignore
        )

    @staticmethod
    def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            float: Average precision.
        """
        extended_recall = np.concatenate(([0.0], recall, [1.0]))
        extended_precision = np.concatenate(([1.0], precision, [0.0]))
        max_accumulated_precision = np.flip(
            np.maximum.accumulate(np.flip(extended_precision))
        )
        interpolated_recall_levels = np.linspace(0, 1, 101)
        interpolated_precision = np.interp(
            interpolated_recall_levels, extended_recall, max_accumulated_precision
        )
        average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
        return average_precision

    @staticmethod
    def _match_detection_batch(
        predictions: np.ndarray, targets: np.ndarray, iou_thresholds: np.ndarray
    ) -> np.ndarray:
        """
        Match predictions with target labels based on IoU levels.

        Args:
            predictions (np.ndarray): Batch prediction. Describes a single image and
                has `shape = (M, 6)` where `M` is the number of detected objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class, conf)` format.
            targets (np.ndarray): Batch target labels. Describes a single image and
                has `shape = (N, 5)` where `N` is the number of ground-truth objects.
                Each row is expected to be in
                `(x_min, y_min, x_max, y_max, class)` format.
            iou_thresholds (np.ndarray): Array contains different IoU thresholds.

        Returns:
            np.ndarray: Matched prediction with target labels result.
        """
        num_predictions, num_iou_levels = predictions.shape[0], iou_thresholds.shape[0]
        correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
        iou = box_iou_batch(targets[:, :4], predictions[:, :4])
        correct_class = targets[:, 4:5] == predictions[:, 4]

        for i, iou_level in enumerate(iou_thresholds):
            matched_indices = np.where((iou >= iou_level) & correct_class)

            if matched_indices[0].shape[0]:
                combined_indices = np.stack(matched_indices, axis=1)
                iou_values = iou[matched_indices][:, None]
                matches = np.hstack([combined_indices, iou_values])

                if matched_indices[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                correct[matches[:, 1].astype(int), i] = True

        return correct

    @staticmethod
    def _average_precisions_per_class(
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
        eps: float = 1e-16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

        Args:
            matches (np.ndarray): True positives.
            prediction_confidence (np.ndarray): Objectness value from 0-1.
            prediction_class_ids (np.ndarray): Predicted object classes.
            true_class_ids (np.ndarray): True object classes.
            eps (float, optional): Small value to prevent division by zero.

        Returns:
            np.ndarray: Average precision for different IoU levels.
        """
        # TODO use print(matches)
        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]

        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]

        average_precisions = np.zeros((num_classes, matches.shape[1]))
        precisions = np.zeros((num_classes, matches.shape[1]))
        recalls = np.zeros((num_classes, matches.shape[1]))

        for class_idx, class_id in enumerate(unique_classes):
            is_class = prediction_class_ids == class_id
            total_true = class_counts[class_idx]
            total_prediction = is_class.sum()

            if total_prediction == 0 or total_true == 0:
                continue

            # print(matches[is_class])
            false_positives = (1 - matches[is_class]).cumsum(0)
            true_positives = matches[is_class].cumsum(0)
            recall = true_positives / (total_true + eps)
            precision = true_positives / (true_positives + false_positives)

            # print(precision)
            # print(recall)
            for iou_level_idx in range(matches.shape[1]):
                average_precisions[class_idx, iou_level_idx] = (
                    MeanAveragePrecision.compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )
                precisions[class_idx, iou_level_idx] = np.mean(
                    precision[-1, iou_level_idx]
                )
                recalls[class_idx, iou_level_idx] = np.mean(recall[-1, iou_level_idx])

        return average_precisions, precisions, recalls
