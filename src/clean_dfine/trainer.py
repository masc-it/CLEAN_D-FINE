"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.amp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
from clean_dfine.arch.dfine import DFINE
from clean_dfine.arch.postprocessor import DFINEPostProcessor
from clean_dfine.config import ExperimentConfig
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam
import torch.nn.functional as F
import torchvision

from clean_dfine.evaluation.evaluator import start_eval
from clean_dfine.model import DFineModel
from clean_dfine.utils import box_ops
import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Dict

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        initial_lr (float): Target learning rate after warmup.
        final_lr (float): Final learning rate after cosine annealing.
        warmup_steps (int): Number of steps for the warmup phase.
        total_steps (int): Total number of training steps.
        warmup_start_lr (float, optional): Starting learning rate for warmup. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        initial_lr,
        final_lr,
        warmup_steps,
        total_steps,
        warmup_start_lr=0.0,
        last_epoch=-1,
    ):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr

        if total_steps < warmup_steps:
            raise ValueError("total_steps must be larger than warmup_steps.")

        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # since last_epoch is initialized to -1
        if current_step <= self.warmup_steps and self.warmup_steps != 0:
            # Linear warmup
            lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * (
                current_step / self.warmup_steps
            )
        elif current_step <= self.total_steps:
            # Cosine annealing
            cosine_steps = current_step - self.warmup_steps
            cosine_total = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_total))
            lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay
        else:
            # After total_steps, keep the final_lr
            lr = self.final_lr

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """Optional: Implement if needed for certain schedulers."""
        return self.get_lr()


def train(
    cfg: ExperimentConfig,
    model: DFINE,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
):
    scaler = GradScaler(cfg.device, enabled=cfg.device != "mps")
    optimizer = Adam(model.parameters(), lr=cfg.lr0, fused=torch.cuda.is_available())

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        alpha=0.25,
        gamma=2.0,
    )

    postprocessor = DFINEPostProcessor(num_classes=cfg.num_classes).to(cfg.device)
    
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        initial_lr=cfg.lr0,
        final_lr=1e-4,
        warmup_steps=len(dataloader_train) * 0.5,
        total_steps=len(dataloader_train) * cfg.num_epochs,
        warmup_start_lr=5e-5,  # Starting from 0
    )
    criterion = CriterionDetection(
        losses=["vfl", "boxes", "focal"],
        weight_dict={
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        alpha=0.75,
        gamma=2.0,
        num_classes=cfg.num_classes,
        matcher=matcher,
    )

    for epoch in range(cfg.num_epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            dataloader_train,
            cfg.device,
            epoch,
        )
        m = DFineModel(model, postprocessor, cfg.img_size, cfg.device)
        print(start_eval(m, dataloader_val))
        model = model.train()
        model.decoder.training = True
        torch.save(m.model.state_dict(), "data/dfine.pt")


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    max_norm: float = 3,
):
    model.train()
    criterion.train()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit="batch")

    for i, (samples, targets) in pbar:
        samples = samples.half().to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]

        global_step = epoch * len(data_loader) + i
        metas = dict(
            epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader)
        )

        optimizer.zero_grad()
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16 if device == "cuda" else None,
            enabled=device != "mps",
        ):
            outputs = model(samples, targets=targets)
            loss = criterion(outputs, targets, **metas)
            # loss = loss_vfl + loss_boxes + loss_focal
            loss = sum(loss.values())
            """ loss = (
                torch.tensor(list(loss.values()), requires_grad=True)
                .sum()
                .to(samples.device)
            ) """

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        if i % 2 == 0:
            # show separate losses
            pbar.set_description(
                f"epoch={epoch}&loss={loss.item():.4f}&norm={norm.item():.2}&lr={scheduler.get_last_lr()[-1]:.5f}"
            )


class CriterionDetection(torch.nn.Module):
    """Default Detection Criterion"""

    def __init__(
        self,
        losses,
        weight_dict,
        num_classes=80,
        alpha=0.75,
        gamma=2.0,
        box_fmt="cxcywh",
        matcher=None,
    ):
        """
        Args:
            losses (list[str]): requested losses, support ['boxes', 'vfl', 'focal']
            weight_dict (dict[str, float)]: corresponding losses weight, including
                ['loss_bbox', 'loss_giou', 'loss_vfl', 'loss_focal']
            box_fmt (str): in box format, 'cxcywh' or 'xyxy'
            matcher (Matcher): matcher used to match source to target
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        assert matcher is not None, ""
        self.matcher = matcher

    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: Dict[Tensor], 'pred_boxes', 'pred_logits', 'meta'.
            targets, List[Dict[str, Tensor]], len(targets) == batch_size.
            kwargs, store other information such as current epoch id.
        Return:
            losses, Dict[str, Tensor]
        """
        matched = self.matcher(outputs, targets)
        indices = matched["indices"]
        num_boxes = self._get_positive_nums(indices)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {
                k: l_dict[k] * self.weight_dict[k]
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)
        # print(losses)
        return losses  # losses["loss_vfl"], losses["loss_bbox"], losses["loss_giou"]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_positive_nums(self, indices):
        # number of positive samples
        num_pos = sum(len(i) for (i, _) in indices)
        num_pos = torch.as_tensor(
            [num_pos], dtype=torch.float32, device=indices[0][0].device
        )
        num_pos = torch.clamp(num_pos, min=1).item()
        return num_pos

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[
            ..., :-1
        ].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.sum() / num_boxes
        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )

        src_boxes = torchvision.ops.box_convert(
            src_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        target_boxes = torchvision.ops.box_convert(
            target_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        iou, _ = box_ops.elementwise_box_iou(src_boxes.detach(), target_boxes)

        src_logits: torch.Tensor = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = iou.to(src_logits.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        src_score = F.sigmoid(src_logits.detach())
        weight = self.alpha * src_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.sum() / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        src_boxes = torchvision.ops.box_convert(
            src_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        target_boxes = torchvision.ops.box_convert(
            target_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_boxes_giou(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        losses = {}
        src_boxes = torchvision.ops.box_convert(
            src_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        target_boxes = torchvision.ops.box_convert(
            target_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "giou": self.loss_boxes_giou,
            "vfl": self.loss_labels_vfl,
            "focal": self.loss_labels_focal,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha)
                * (out_prob**self.gamma)
                * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(out_bbox), box_ops.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (
                        torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64),
                    )
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
