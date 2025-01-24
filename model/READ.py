# --------------------------------------------------------
# See, Say, and Segment: Teaching LMMs to Overcome False Premises (CVPR 2024).
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Tsung-Han Wu*, Giscard Biamby*, David M. Chan et al.
# --------------------------------------------------------
# Reasoning to Attend: Try to Understand How <SEG> Token Works
# Modified by Rui Qian
# --------------------------------------------------------
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from model.llava.model import *
from model.llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from .llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .segment_anything import build_sam_vit_h
from model.segment_anything import sam_model_registry, SamPredictor
from model.segment_anything.utils.transforms import ResizeLongestSide
import glob
import os
import numpy as np
from PIL import Image
import cv2
from  matplotlib import pyplot as plt

debug = False
if debug:
    # Init original SAM 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_checkpoint = "../dataset_sesame/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,  # α参数，控制正负样本的权重
    gamma: float = 2.0,   # γ参数，控制难易样本的衰减
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The raw predictions (logits) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks, used for normalizing the loss.
        alpha: Weighting factor for balancing positive and negative samples.
        gamma: Focusing parameter to decrease the relative loss for well-classified examples.

    Returns:
        Loss tensor
    """
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)  
    focal_factor = (1 - p_t) ** gamma 
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_factor * ce_loss
    loss = loss.flatten(1, 2) 
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss

class ReadMetaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_read_modules(self.config)

    def initialize_read_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class ReadModel(ReadMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(ReadModel, self).__init__(config, **kwargs)
        # self.config = config
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

class READForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):  
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.default_im_start_token_idx = kwargs.pop("default_im_start_token_idx")
        super(READForCausalLM, self).__init__(config)
        self.model = ReadModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
 
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)

        return self.model_forward(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs

    def generate_pred_masks(
            self, 
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            image_path,
            point_coords,
            point_labels,
            masks_list,
            conversation_list
        ):
          
        multimask_output = False
        pred_masks = []

        def get_preprocess_shape(
            oldh: int, oldw: int, long_side_length: int
        ):
            """
            Compute the output size given input size and target long side length.
            """
            scale = long_side_length * 1.0 / max(oldh, oldw)
            newh, neww = oldh * scale, oldw * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            return (newh, neww)
    
        def apply_coords(
            coords, 
            original_size,
        ) -> np.ndarray:
            """
            Expects a numpy array of length 2 in the final dimension. Requires the
            original image size in (H, W) format.
            """
            if coords.numel() ==0: return coords
            old_h, old_w = original_size
            new_h, new_w = get_preprocess_shape(
                original_size[0], original_size[1], 1024
            )
            coords = coords.clone().float()
            coords[..., 0] = coords[..., 0] * (new_w / old_w)
            coords[..., 1] = coords[..., 1] * (new_h / old_h)
            return coords
        
        for i in range(len(pred_embeddings)):
            # For inference (testing) mode only
            if pred_embeddings[i] is None:
                pred_mask = torch.zeros(sam_mask_shape_list[i][1]).to(image_embeddings.device).int()
                pred_masks.append(pred_mask)
                continue
            
            point_coords_ = apply_coords(point_coords[i], sam_mask_shape_list[i][1])
            point_coords_ = torch.as_tensor(
                point_coords_, dtype=torch.float, device=self.device
            )
            point_labels_ = torch.as_tensor(
                point_labels[i], dtype=torch.int, device=self.device
            )
            points = (point_coords_, point_labels_)
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=points, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1)
            )             
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                image_path =None # image_path[i]
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks, input_size=sam_mask_shape_list[i][0], original_size=sam_mask_shape_list[i][1]
            )
            pred_masks.append(pred_mask[:, 0])
            
            debug = False
            if debug:
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1)
                )  
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                    image_path = image_path[i]
                )
                pred_text_embed_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks, input_size=sam_mask_shape_list[i][0], original_size=sam_mask_shape_list[i][1]
                )
            
            debug = False
            if debug:# only support batch size = 1, set batch size=1 to debug
                pil_img = Image.open(image_path[i])
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                point_coords[i] = point_coords[i].detach().cpu().numpy() # tmpmtmp

                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # sam_checkpoint = "../dataset_sesame/sam_vit_h_4b8939.pth"
                # model_type = "vit_h"
                # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                # sam.to(device=device)
                # predictor = SamPredictor(sam)
                
                # predictor.set_image(np.array(pil_img))
                cv2_img = cv2.imread(image_path[i])
                predictor.set_image(cv2_img)
                mask_sam = []
                for m in range(point_labels[i].shape[0]):
                    masks, scores, logits = predictor.predict(point_labels=point_labels[i][m], \
                                    point_coords=point_coords[i][m], multimask_output=True)
                    mask = masks[np.argmax(scores)]
                    mask = mask.astype('uint8')
                    mask_sam.append(torch.tensor(mask).to(pred_mask)[None,...])
                
                pred_mask_ = pred_mask[:, 0,...].clone().detach().cpu().numpy()
                # pred_text_embed_mask_ = pred_text_embed_mask[:, 0,...].clone().detach().cpu().numpy()
                for bs in range(pred_mask_.shape[0]):
                    # Visualize the results
                    vis = cv2_img.copy()
                    vis[pred_text_embed_mask_[bs] > 0] = vis[pred_text_embed_mask_[bs] > 0] // 2 + np.array([0, 0, 255], dtype=np.uint8) // 2
                    for j, [x, y] in enumerate(point_coords[i][bs].astype(int).tolist()):
                        cv2.circle(vis, (x, y), 4, (255, 255, 255), 3)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255) if point_labels[0][bs][j] == 1 else (255, 0, 0), 3)
                    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                    plt.subplot(2, 2, 1) 
                    plt.imshow(vis, cmap='viridis')
                    plt.title('s2p(Ours:text embed) pred Image')

                    vis = cv2_img.copy()
                    vis[pred_mask_[bs] > 0] = vis[pred_mask_[bs] > 0] // 2 + np.array([0, 0, 255], dtype=np.uint8) // 2
                    for j, [x, y] in enumerate(point_coords[i][bs].astype(int).tolist()):
                        cv2.circle(vis, (x, y), 4, (255, 255, 255), 3)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255) if point_labels[0][bs][j] == 1 else (255, 0, 0), 3)
                    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                    plt.subplot(2, 2, 2) 
                    plt.imshow(vis, cmap='viridis')
                    plt.title('s2p(Ours:points embed) pred Image')

                    vis = cv2_img.copy()
                    vis[mask_sam[bs] > 0] = vis[mask_sam[bs] > 0] // 2 + np.array([0, 0, 255], dtype=np.uint8) // 2
                    for j, [x, y] in enumerate(point_coords[i][bs].astype(int).tolist()):
                        cv2.circle(vis, (x, y), 4, (255, 255, 255), 3)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255) if point_labels[i][bs][j] == 1 else (255, 0, 0), 3)
                    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                    plt.subplot(2, 2, 3) 
                    plt.imshow(vis, cmap='viridis')
                    plt.title('sam(points embed) pred Image')
                    plt.show()
                   
                    mask_ = masks_list[i].detach().cpu().numpy()
                    vis = cv2_img.copy()
                    vis[mask_[bs] > 0] = vis[mask_[bs] > 0] // 2 + np.array([0, 255, 0], dtype=np.uint8) // 2
                    for j, [x, y] in enumerate(point_coords[i][bs].astype(int).tolist()):
                        cv2.circle(vis, (x, y), 4, (255, 255, 255), 3)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255) if point_labels[i][bs][j] == 1 else (255, 0, 0), 3)
                    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                    plt.subplot(2, 2, 4) 
                    plt.imshow(vis, cmap='viridis')
                    plt.title('gt Image')
                    import textwrap
                    caption = conversation_list[bs].split("the human's questions.")[-1]
                    caption = "\n".join(textwrap.wrap(caption, width=120))
                    plt.suptitle(caption, fontsize=8)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()
        return pred_masks

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        sam_mask_shape_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        
        batch_size = len(sam_mask_shape_list)
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        # HACK: padding numer-of-token-per-image in total 
        vision_tower = self.get_vision_tower()
        num_tokens_per_image = vision_tower.num_patches
        padding_left = torch.zeros(
            seg_token_mask.shape[0],
            num_tokens_per_image - 1,
            dtype=seg_token_mask.dtype,
            device=seg_token_mask.device,
        )
        padding_right = torch.zeros(
            seg_token_mask.shape[0],
            1,
            dtype=seg_token_mask.dtype,
            device=seg_token_mask.device,
        )
        seg_token_mask = torch.cat(
            [padding_left, seg_token_mask, padding_right],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                torch.cuda.empty_cache()

            output_hidden_states = output_i.hidden_states
            output = None
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        assert len(self.model.text_hidden_fcs) == 1
        pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states)[
            seg_token_mask
        ]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset],
            dim=0,
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        points_list, labels_list = self.similarity_as_points(
            output_hidden_states, 
            seg_token_mask, 
            offset,
            input_ids,
            masks_list,
            sam_mask_shape_list,
            images_clip,
            kwargs['image_paths']
        )
        points_list_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            points_list_.append(points_list[start_i:end_i])
        points_list = points_list_

        labels_list_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            labels_list_.append(labels_list[start_i:end_i])
        labels_list = labels_list_
        
        # Run SAM
        image_embeddings = self.get_visual_embs(images)
        pred_masks = self.generate_pred_masks(
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            kwargs['image_paths'],
            points_list,
            labels_list,
            masks_list,
            kwargs['conversation_list']
        )
        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss 

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    
    def similarity_as_points(
            self, 
            output_hidden_states, 
            seg_token_mask,
            offset,
            input_ids, 
            masks_list,
            sam_mask_shape_list,
            images_clip,
            image_path
    ):
        
        def get_similarity_map(sm, shape):
            # min-max norm
            sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
            # reshape
            side = int(sm.shape[1] ** 0.5) # square output
            sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
            # interpolate
            sm = sm.to(torch.float32)

            target_size = 336
            h, w = shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
            pad_h = (new_h - target_size) // 2
            pad_w = (new_w - target_size) // 2
            padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
            sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
            sm = sm.permute(0, 2, 3, 1)
            return sm

        def compute_similarity_map(
            image_features, 
            text_features, 
            redundant_feats=None
        ):  
            """see also: https://github.com/rui-qian/CLIP_Surgery/blob/master/demo.py"""
            if redundant_feats != None:
                similarity = image_features @ (text_features - redundant_feats).t()
            else:
                image_features = image_features.clone()
                text_features = text_features.clone()
                prob = image_features[:, :1, :] @ text_features.t()
                prob = (prob * 2).softmax(-1)
                w = prob / prob.mean(-1, keepdim=True)
                b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], \
                    image_features.shape[1], image_features.shape[2]
                feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
                feats *= w.reshape(1, 1, n_t, 1)
                # sum the element-wise multiplied features as cosine similarity
                similarity = feats.sum(-1)
            return similarity
        
        images_size_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_size_list.extend([sam_mask_shape_list[i][1]] * (end_i - start_i))

        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        image_embedding_tokens = output_hidden_states[seg_token_counts==1]
        seg_embedding_tokens = output_hidden_states[seg_token_mask]
                
        points_list, labels_list = [], []
        for bs in range(len(image_embedding_tokens)):
            default_im_start_token_idx = torch.where(
                input_ids==self.default_im_start_token_idx
            )[1][0].item()

            similarity = compute_similarity_map(
                image_embedding_tokens [ 
                    bs: bs+1, 
                    default_im_start_token_idx + 1: default_im_start_token_idx + 1 \
                    + self.get_vision_tower().num_patches, :
                ],
                seg_embedding_tokens[bs: bs + 1, ...]
            )
            points1, labels1 = self.similarity_map_to_points(similarity[0, :, 0], images_size_list[bs], t=0.8)
            points_list.append(points1)
            labels_list.append(labels1)

            debug = False
            if debug:
                masks_list_ = []
                for i  in masks_list:
                    masks_list_.extend(i)
                masks_list = masks_list_

                points = points1.detach().cpu().numpy()
                labels = labels1.clone()
                images_list = []
                for i in range(len(offset) - 1):
                    # pil_img = Image.open(image_path[i])
                    # cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2_img = cv2.imread(image_path[i]) #[:, :, ::-1]
                    # plt.imshow(cv2_img)
                    # plt.show()
                    start_i, end_i = offset[i], offset[i + 1]
                    images_list.extend([cv2_img] * (end_i - start_i))

                images_clip_ = self.encode_images(images_clip)
                similarity_clip_ = compute_similarity_map(
                    images_clip_[bs: bs + 1, ...],
                    seg_embedding_tokens[bs: bs + 1, ...]
                )
                similarity_clip_map = get_similarity_map(similarity_clip_, images_list[bs].shape[:2])
                vis = (similarity_clip_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = images_list[bs] * 0.3 + vis * 0.7
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                plt.subplot(2, 2, 1) 
                plt.imshow(vis, cmap='viridis')
                plt.title('similarity_clip_map')

                similarity_map = get_similarity_map(similarity, images_list[bs].shape[:2])
                vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = images_list[bs] * 0.3 + vis * 0.7
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                plt.subplot(2, 2, 2) 
                plt.imshow(vis, cmap='viridis')
                plt.title('similarity_map')

                predictor.set_image(images_list[bs])
                masks, scores, logits = predictor.predict(point_labels=labels, \
                          point_coords=np.array(points), multimask_output=True)
                sam_mask = masks[np.argmax(scores)]
                sam_mask = sam_mask.astype('uint8')
                vis = images_list[bs].copy()
                vis[sam_mask > 0] = vis[sam_mask > 0] // 2 + np.array([0, 0, 255], dtype=np.uint8) // 2
                for i, [x, y] in enumerate(points):
                    x, y = int(x), int(y)
                    cv2.circle(vis, (x, y), 3, (255, 255, 255), 3)
                    cv2.circle(vis, (x, y), 2, (0, 0, 255) if labels[i] == 1 else (255, 0, 0), 3)
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                plt.subplot(2, 2, 3) 
                plt.imshow(vis, cmap='viridis')
                plt.title('sam pred')

                mask_gt = masks_list[bs].detach().cpu().numpy()
                vis = images_list[bs].copy()
                vis[mask_gt > 0] = vis[mask_gt > 0] // 2 + np.array([0, 255, 0], dtype=np.uint8) // 2
                for i, [x, y] in enumerate(points):
                    x, y = int(x), int(y)
                    cv2.circle(vis, (x, y), 4, (255, 255, 255), 3)
                    cv2.circle(vis, (x, y), 2, (0, 0, 255) if labels[i] == 1 else (255, 0, 0), 3)
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                plt.subplot(2, 2, 4) 
                plt.imshow(vis, cmap='viridis')
                plt.title('gt mask')
                plt.show()
        
        if len(points_list) > 0:
           stacked_points = torch.stack(points_list, dim=0)
           stacked_labels = torch.stack(labels_list, dim=0)
        else:
            stacked_points = torch.empty(0, dtype=torch.long)
            stacked_labels = torch.empty(0, dtype=torch.long)
        return stacked_points, stacked_labels
    
    # uncomment for READ-13B ReasonSeg test set
    # def similarity_map_to_points(self, sm, shape, t=0.8, down_sample=1):
    # uncomment for READ-7B ReasonSeg val/test set  
    def similarity_map_to_points(self, sm, shape, t=0.8, down_sample=2):
        def get_similarity_map(sm, shape):
            # min-max norm
            sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
            # reshape
            side = int(sm.shape[1] ** 0.5) # square output
            sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
            # interpolate
            sm = sm.to(torch.float32)

            target_size = 336
            h, w = shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
            pad_h = (new_h - target_size) // 2
            pad_w = (new_w - target_size) // 2
            padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
            sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
            sm = sm.permute(0, 2, 3, 1)
            return sm
        
        def Discrete_to_Continuous(sm, selected_points):
            h, w = sm.shape
            softmax_probs = F.softmax(sm.view(-1), dim=0) 
            grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
            grid_x = grid_x.to(softmax_probs.device).float().view(-1) 
            grid_y = grid_y.to(softmax_probs.device).float().view(-1)

            selected_x = selected_points[:, 0]
            selected_y = selected_points[:, 1]

            continuous_coordinates = []
            
            for x, y in zip(selected_x, selected_y):
        
                distances = ((grid_x - x) ** 2 + (grid_y - y) ** 2)
                weight = torch.exp(-distances)  

                final_weights = weight * softmax_probs
                final_weights = final_weights / final_weights.sum()  

                continuous_x = (grid_x * final_weights).sum()
                continuous_y = (grid_y * final_weights).sum()

                continuous_coordinates.append(torch.stack([continuous_x, continuous_y]))
            return torch.stack(continuous_coordinates)
        
        origin_sm = get_similarity_map(sm[None, ..., None], shape)
        # sm shape N_t
        side = int(sm.shape[0] ** 0.5)
        sm = sm.reshape(1, 1, side, side)
        # down sample to smooth results
        down_side = side // down_sample
        sm = sm.to(torch.float32)
        sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
        sm = (sm - sm.min()) / (sm.max() - sm.min())
        
        target_size = down_side #12
        ori_h, ori_w = shape
        scale = target_size / min(ori_h, ori_w)
        new_h, new_w = int(ori_h * scale), int(ori_w * scale)
        pad_h = (new_h - target_size) // 2
        pad_w = (new_w - target_size) // 2
        #sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
        sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=-1)
        h, w = sm.shape
        scale_h = float(shape[0]) / h
        scale_w = float(shape[1]) / w
        sm = sm.reshape(-1)
        #sm = (sm - sm.min()) / (sm.max() - sm.min())
       
        # # uncomment for READ-7B ReasonSeg val/test set 
        num_points = 30
        t_pos = 0.8 
        t_neg = 0.2 
        # # uncomment for READ-13B ReasonSeg test set
        # num_points = 60
        # t_pos = 0.6
        # t_neg = 0.2

        #mean_val = sm.mean().item()
        #std_val = sm.std().item()
        #std_factor = 1.28
        #t_pos = mean_val + std_val * std_factor
        #t_neg = mean_val - std_val * std_factor

        # (t >= 0.8)，label = 1
        pos_indices = (sm >= t_pos).nonzero(as_tuple=False).squeeze()
        if pos_indices.dim() == 0:
            pos_indices = pos_indices.unsqueeze(0)
        pos_values = sm[pos_indices]
        sorted_pos_indices = pos_indices[torch.argsort(pos_values, descending=True)]  
        num_pos = sorted_pos_indices.numel()

        # (t <= 0.2)，label = 0
        neg_indices = ((sm >= 0) & (sm <= t_neg)).nonzero(as_tuple=False).squeeze()
        if neg_indices.dim() == 0:
            neg_indices = neg_indices.unsqueeze(0)
        neg_values = sm[neg_indices]
        sorted_neg_indices = neg_indices[torch.argsort(neg_values)] 
        num_neg = sorted_neg_indices.numel()

        #(0.2 < t < 0.8)，label = -1
        neutral_indices = ((sm > t_neg) & (sm < t_pos)).nonzero(as_tuple=False).squeeze()
        if neutral_indices.dim() == 0:
            neutral_indices = neutral_indices.unsqueeze(0)
        num_neutral = neutral_indices.numel()


        selected_indices = torch.cat([sorted_pos_indices.to(sm.device), sorted_neg_indices.to(sm.device), neutral_indices.to(sm.device)], dim=0)
        selected_indices = selected_indices[:num_points]

        if selected_indices.dim() == 0:
            selected_indices = selected_indices.unsqueeze(0)

        labels = torch.full((selected_indices.numel(),), -1, dtype=torch.long) 

        points = []
        #labels = []
        # Generate points and labels based on selected indices
        bound_w = torch.tensor(shape[1] - 1).to(device=sm.device, dtype=torch.float32)
        bound_h = torch.tensor(shape[0] - 1).to(device=sm.device, dtype=torch.float32)
        for i, idx in enumerate(selected_indices):
            x = min((idx % w + 0.5) * scale_w, bound_w)  # +0.5 to center
            y = min((idx // w + 0.5) * scale_h, bound_h)
            points.append([int(x.item()), int(y.item())])
            if idx in pos_indices:
               labels[i] = 1
            if idx in neg_indices:
               labels[i] = 0

        points = torch.tensor(points)
        points = Discrete_to_Continuous(origin_sm[0, ..., 0], points)   
        return points, labels

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        sam_mask_shape_list,
        max_new_tokens=32,
    ):
        with torch.inference_mode():

            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=0.2
            )

            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            #first_true_index = torch.argmax(seg_token_mask.clone().int())
            # 将第一个 True 后的所有 True 设置为 False
            #seg_token_mask[:, first_true_index+1:] = False

            # HACK: padding numer-of-token-per-image in total 
            vision_tower = self.get_vision_tower()
            num_tokens_per_image = vision_tower.num_patches
            padding_left = torch.zeros(
                seg_token_mask.shape[0],
                num_tokens_per_image - 1,
                dtype=seg_token_mask.dtype,
                device=seg_token_mask.device,
            )
            seg_token_mask = torch.cat(
                [padding_left, seg_token_mask],
                dim=1,
            )
            assert len(self.model.text_hidden_fcs) == 1
            output_hidden_states = output_hidden_states.to(seg_token_mask.device)
            pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states)
            pred_embeddings = pred_embeddings.to(seg_token_mask.device)
            pred_embeddings = pred_embeddings[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset],
                dim=0,
            )

            pred_embeddings_ = []
            object_presence = []
            for i in range(len(seg_token_offset) - 1):
                if seg_token_counts[i] == 0:
                    pred_embeddings_.append(None)
                    object_presence.append(False)
                else:
                    start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                    pred_embeddings_.append(pred_embeddings[start_i:end_i])
                    object_presence.append(True)
           
            pred_embeddings = pred_embeddings_

            # Run SAM
            image_embeddings = self.get_visual_embs(images)
            
            image_embedding_tokens = output_hidden_states[seg_token_counts==1]
            seg_embedding_tokens = output_hidden_states[seg_token_mask]
            default_im_start_token_idx = torch.where(input_ids==32001)[1][0].item()
            points_list, labels_list = [], []
            for bs in range(len(image_embedding_tokens)):
                similarity = self.compute_similarity_map(
                    image_embedding_tokens[ 
                        bs: bs+1, 
                        default_im_start_token_idx + 1: default_im_start_token_idx + 1 \
                        + self.get_vision_tower().num_patches, :
                    ],
                    seg_embedding_tokens[bs: bs + 1, ...]
                )
                points1, labels1 = self.similarity_map_to_points(similarity[0, :, 0], sam_mask_shape_list[0][1], t=0.8)
                points_list.append(points1[None,...])
                labels_list.append(labels1[None,...])

            pred_masks = self.generate_pred_masks(
            pred_embeddings, 
            image_embeddings, 
            sam_mask_shape_list, 
            image_path=None,
            point_coords=points_list,
            point_labels=labels_list,
            masks_list=None,
            conversation_list=None
        )  
            # pred_masks = self.generate_pred_masks(pred_embeddings, image_embeddings, sam_mask_shape_list)
            # Post processing for inference
            output_pred_masks = []
            for i, pred_mask in enumerate(pred_masks):
                if pred_embeddings[i] is not None:
                    pred_mask = (pred_mask[0] > 0).int()
                    if pred_mask.sum() == 0:
                        object_presence[i] = False
                    output_pred_masks.append(pred_mask)
                else:
                    output_pred_masks.append(pred_mask)
        
        return output_ids, output_pred_masks, object_presence
    
    def compute_similarity_map(
            self,
            image_features, 
            text_features, 
            redundant_feats=None
        ):
            if redundant_feats != None:
                similarity = image_features @ (text_features - redundant_feats).t()
            else:
                image_features = image_features.clone()
                text_features = text_features.clone()
                # weights to restrain influence of obvious classes on others
                prob = image_features[:, :1, :] @ text_features.t()
                prob = (prob * 2).softmax(-1)
                w = prob / prob.mean(-1, keepdim=True)
                b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], \
                    image_features.shape[1], image_features.shape[2]
                feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
                feats *= w.reshape(1, 1, n_t, 1)
                # sum the element-wise multiplied features as cosine similarity
                similarity = feats.sum(-1)
            return similarity

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []

            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings


def load_pretrained_model_READ(
    model_path,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs["device_map"] = device_map
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    default_im_start_token_idx = tokenizer(DEFAULT_IM_START_TOKEN, add_special_tokens=False).input_ids[0]
    kwargs["seg_token_idx"] = seg_token_idx
    kwargs["default_im_start_token_idx"] = default_im_start_token_idx
    # model = SESAMEForCausalLM.from_pretrained(
    #     model_path, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs
    # )
    model = READForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,**kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.model_max_length = kwargs.get("model_max_length", None)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))
    if "training" in kwargs and kwargs["training"] is True:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    # vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=model.dtype)
    # image_processor = vision_tower.image_processor

    if hasattr(model.config, "model_max_length"):
        context_len = model.config.model_max_length
    else:
        context_len = 2048
    
    return tokenizer, model, vision_tower, context_len

def init_READ_model(args, model_args):
    #Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        # padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    args.default_im_start_token_idx = tokenizer(DEFAULT_IM_START_TOKEN, add_special_tokens=False).input_ids[0]
    model_args["seg_token_idx"] = args.seg_token_idx
    model_args["default_im_start_token_idx"] = args.default_im_start_token_idx
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16  
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = READForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True,**model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.use_released_param: 
        pass # use released param by default, otherwise initialize SAM.
    else:
        model.get_model().initialize_read_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Configure LoRA if applicable
    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            exclude_list = ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"]
            for name, module in model.named_modules():
                if isinstance(module, cls) and not any(x in name for x in exclude_list) \
                    and any([x in name for x in lora_target_modules]):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    for n, p in model.named_parameters():
        if any(part in n for part in trainable_parts):
            p.requires_grad = True
    return tokenizer, model, vision_tower

