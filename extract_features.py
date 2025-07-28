import argparse
import glob
import os
import shutil
import logging
import random
import warnings
import time
import io
import sys
import cv2
import warnings
import torch

import numpy as np
import albumentations as A
import torch.nn.functional as F

from torch import nn, Tensor
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from icecream import ic
from PIL import Image
from torchvision import transforms as torchtrans
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from albumentations.pytorch.transforms import ToTensorV2

# IMPORT FOR MODEL
from collections import OrderedDict

from torchvision.ops import nms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference,
    keypointrcnn_loss,
    keypointrcnn_inference,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNConvFCHead,
    _default_anchorgen,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    RoIHeads,
    RegionProposalNetwork, RPNHead,
    GeneralizedRCNNTransform,
    MultiScaleRoIAlign,
    TwoMLPHead,
    _ovewrite_value_param,
    misc_nn_ops,
    overwrite_eps,
)

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)
from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    RPNHead
)
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

#==================== FUNCTION=============================
#-- Identity class
class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, X):
        return X
    

#-- Extraction Model Class
class ExtractingRoiHead(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, mask_roi_pool=None, mask_head=None, mask_predictor=None, keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None):
        super().__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, mask_roi_pool, mask_head, mask_predictor, keypoint_roi_pool, keypoint_head, keypoint_predictor)

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")
        # ic(features)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # ic(box_features.shape)
        box_features = self.box_head(box_features)

        return box_features, proposals

class ExtractingGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__(backbone, rpn, roi_heads, transform)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        features = self.backbone(images.tensors)
        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        # ic(proposals.shape)
        # ic(features.shape)
        box_features, proposals = self.roi_heads(features, proposals, images.image_sizes, targets)
        return box_features, features, proposals, proposal_losses, images, original_image_sizes


#-- Predictor Model Class
class PredictingRoiHead(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, mask_roi_pool=None, mask_head=None, mask_predictor=None, keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None):
        super().__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, mask_roi_pool, mask_head, mask_predictor, keypoint_roi_pool, keypoint_head, keypoint_predictor)

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        box_features,
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]

    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        labels = None
        regression_targets = None
        matched_idxs = None

        ##
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        # Trash 1
        return result, losses, class_logits, box_regression

class PredictingGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__(backbone, rpn, roi_heads, transform)

    def forward(self, images, features, proposals, box_features, proposal_losses, original_image_sizes, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        detections, detector_losses, class_logits, box_regression = self.roi_heads(features, proposals, images.image_sizes, box_features, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections), class_logits, box_regression


#-- Faster RCNN Model
class CustomExtractingFRCNN(ExtractingGeneralizedRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = ExtractingRoiHead(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)
        self.roi_heads = roi_heads
        # self.mod_roi_heads()

    def mod_roi_heads(self):
        self.roi_heads.box_predictor = Identity()
        self.roi_heads.box_head.fc7 = Identity()

class CustomPredictingFRCNN(PredictingGeneralizedRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.00, # Select all boxes
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = PredictingRoiHead(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)
        self.roi_heads = roi_heads
        # self.mod_roi_heads()

    def mod_roi_heads(self):
        module_names = [pair[0] for pair in list(self.named_children())]
        identity_module_names = module_names[1:3]
        for name in identity_module_names:
            setattr(self, name, Identity())
        self.roi_heads.box_roi_pool = Identity()
        self.roi_heads.box_head.fc6 = Identity()


#-- Extract features
class FeatureExtractor:
    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    )
    CONFIG_URL = (
        "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    )
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser()
        self.device = self.args.device
        os.makedirs(self.args.output_folder, exist_ok=True)
        self.load_model()

    def load_model(self):
        weights_path = {
            "pretrained": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
            "weights_backbone": "ResNet50_Weights.IMAGENET1K_V1",
        }
        weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights_path['pretrained'])
        weights_backbone = ResNet50_Weights.verify(weights_path['weights_backbone'])
        is_trained = False
        trainable_backbone_layers = _validate_trainable_layers(None, None, 5, 3)
        trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
        norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

        backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=norm_layer)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        self.model_extracting = CustomExtractingFRCNN(
            backbone,
            num_classes=91
        )
        self.model_predicting = CustomPredictingFRCNN(
            backbone,
            num_classes=91
        )
        if weights is not None:
            self.model_extracting.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
            self.model_predicting.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
            if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
                    overwrite_eps(self.model_extracting, 0.0)
                    overwrite_eps(self.model_predicting, 0.0)
        self.model_extracting = self.model_extracting.to(self.device)
        self.model_predicting = self.model_predicting.to(self.device)
        


    def get_parser(self):
        # Argument Parser setup
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_file", default=None, type=str, help="Detectron model file")
        parser.add_argument("--config_file", default=None, type=str, help="Detectron config file")
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--num_features", type=int, default=100, help="Number of features to extract.")
        parser.add_argument("--output_folder", type=str, default="./output", help="Output folder")
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument("--feature_name", type=str, help="The name of the feature to extract", default="fc6")
        parser.add_argument("--confidence_threshold", type=float, default=0, help="Threshold of detection confidence above which boxes will be selected")
        parser.add_argument("--background", action="store_true", help="The model will output predictions for the background class when set")
        parser.add_argument("--device", type=str, default="cuda:5", help="Threshold of detection confidence above which boxes will be selected")

        # Parse arguments
        args = parser.parse_args()
        return args

    def _image_transform(self, path):
        img = cv2.imread(path,1)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size. If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max
        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        im_info = {
            "width": im_width,
            "height": im_height
        }
        return img, im_scale, im_info


    def _process_feature_extraction(self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0):
        # Hyperparams
        batch_size = len(output['proposals'])
        n_boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in output['proposals']]

        # Split by image
        score_list = output['scores'].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        cur_device = score_list[0].device
        feats = output['features'].split(n_boxes_per_image)

        # Get features
        feat_list = []
        info_list = []
        for i in range(batch_size):
            dets = output["proposals"][i]/ im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep]) &
                    (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep], max_conf[keep]
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[:self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[:self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output["proposals"][i][keep_boxes]/ im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list
    
    def get_features_results(self, list_images):
        # Extracting
        self.model_extracting.eval()
        box_features, features, proposals, proposal_losses, images, original_image_sizes = self.model_extracting(list_images)

        # Predictor
        self.model_predicting.eval()
        result, class_logits, box_regression = self.model_predicting(
            images=images,
            features=features,
            proposals=proposals,
            box_features=box_features,
            proposal_losses=proposal_losses,
            original_image_sizes=original_image_sizes,
        )

        # Pred_boxes
        pred_boxes_logits = self.model_predicting.roi_heads.box_coder.decode(box_regression, proposals)

        # Return
        return box_features, class_logits, proposals, pred_boxes_logits


    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        # Output
        outputs = []
        error_list = []
        for idx, image_path in enumerate(image_paths):
            try:
                im, im_scale, im_info = self._image_transform(image_path)
            except:
                error_list.append(idx)
                continue
            img_tensor.append(torch.tensor(im).to(self.device))
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # If all image in batch is error
        if len(img_tensor) == 0:
            return (None, None), error_list

        with torch.inference_mode():
            # img_tensor = torch.concatenate(img_tensor).to(self.device)
            box_features, class_logits, proposals, pred_boxes_logits = self.get_features_results(
                list_images=img_tensor
            )
            output = {
                'features': box_features,
                'proposals': proposals,
                'scores': class_logits,
                'boxes': pred_boxes_logits,
            }

        output_list = self._process_feature_extraction(
            output, im_scales, im_infos, self.args.feature_name,
            self.args.confidence_threshold
        )
        return output_list, error_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            os.path.join(self.args.output_folder, "features", file_base_name), feature.cpu().numpy()
        )
        np.save(os.path.join(self.args.output_folder, "info", info_file_base_name), info)
        return file_base_name

    def extract_features(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            (features, infos), error_list = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            list_extracted_images = os.listdir(os.path.join(self.args.output_folder, "features"))
            files = glob.glob(os.path.join(image_dir, "*.png"))
            ic(len(files))
            ic(os.path.join(image_dir, "*.png"))
            ids = [os.path.basename(file).split(".")[0] for file in files]
            # Filter files
            files = [
                file
                for file, id in zip(files, ids)
                if f"{id}.npy" not in list_extracted_images
            ]
            pbar = tqdm(enumerate(self._chunks(files, self.args.batch_size)), total=len(files) // self.args.batch_size)
            for iter, chunk in pbar:
                (features, infos), error_list = self.get_detectron_features(chunk)
                for idx, file_name in enumerate(chunk):
                    if idx in error_list:
                        continue
                    name = self._save_feature(file_name, features[idx-len(error_list)], infos[idx-len(error_list)])
                    pbar.set_postfix(Saving=name, refresh=True)

#======================= MAIN ==================
np.float = float
if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()



