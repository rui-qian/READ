# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        image_path = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_path = image_path
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_path = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings: [1, C, H, W], tokens: [B, N, C]
        # dense_prompt_embeddings: [B, C, H, W]
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        debug = False
        if debug:
            hs_hook = hs.clone()
            src_hook = src.clone()
            from PIL import Image
            import cv2
            from  matplotlib import pyplot as plt
            import numpy as np
            # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/scene0104_00_0.jpg" #chair
            # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/206674234_4cb520b13d_o.jpg"
            # image_path = "../dataset_sesame/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg" #deer
            # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/2881277421_416273151c_o.jpg"
            # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/536167533_22228a08df_o.jpg"
            pil_img = Image.open(image_path)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # 1x4096x256 1x256
            if hs_hook.numel():
                similarity = compute_similarity_map(src_hook[0:1, ...], hs_hook[0, -2:-1, :])
                similarity_map = get_similarity_map(similarity, cv2_img.shape[:2])
                vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
                
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = cv2_img * 0.3 + vis * 0.7
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                plt.subplot(1, 1, 1) 
                plt.imshow(vis, cmap='viridis')
                plt.title('similarity_sam_map')
                plt.show()

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, self.num_mask_tokens, h, w
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# for debug use
def compute_similarity_map(
    image_features, 
    text_features, 
    redundant_feats=None
):  
    """see also: https://github.com/rui-qian/CLIP_Surgery/blob/master/demo.py"""
    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()
    else:
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

def get_similarity_map(sm, shape):
    
    target_length = 1024
    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
    # reshape
    side = int(sm.shape[1] ** 0.5) # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    # interpolate
    sm = sm.to(torch.float32)
    sm = torch.nn.functional.interpolate(sm, (target_length, target_length), mode='bilinear')
    
    oldh, oldw = shape
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    sm = sm[:, :, 0:newh, 0:neww]
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm
# for debug use


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
