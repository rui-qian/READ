import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            
            debug = False
            if debug:
                visualize()

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def visualize():
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import cv2
    from  matplotlib import pyplot as plt
    import numpy as np
    processor = CLIPProcessor.from_pretrained("../dataset_sesame/clip-vit-large-patch14-336")
    text = 'antler'
    image_path = "../dataset_sesame/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg"
    # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/scene0104_00_0.jpg" #chair
    # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/206674234_4cb520b13d_o.jpg"
    # image_path = "../dataset_sesame/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg" #deer
    # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/2881277421_416273151c_o.jpg"
    # image_path = "../dataset_sesame/reason_seg/ReasonSeg/val/536167533_22228a08df_o.jpg"
    image = Image.open(image_path)
    import pdb;pdb.set_trace()
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device=self.device)
    model = CLIPModel.from_pretrained("../dataset_sesame/clip-vit-large-patch14-336").to(device=self.device, dtype=self.dtype)
    with torch.no_grad():
        outputs = model.vision_model(
            inputs['pixel_values'].to(device=self.device, dtype=self.dtype), 
            output_hidden_states=True
        )
        patch_embeddings = outputs.hidden_states[-1]  # 最后一层的 patch 特征
        text_features = model.get_text_features(
            input_ids=inputs['input_ids'].to(device=self.device), 
            attention_mask=inputs['attention_mask'].to(device=self.device)
        )

    # Patch embedding: 去掉 CLS token，剩下的就是 patch embedding
    # patch_embeddings = patch_embeddings[:, 1:, :]  # 去掉第一个 CLS token
    patch_embeddings = patch_embeddings[:, :, :]  # 去掉第一个 CLS token
    patch_embeddings = model.visual_projection(patch_embeddings)

    # 计算文本特征和 patch 特征的相似度
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
    patch_embeddings = patch_embeddings / patch_embeddings.norm(dim=-1, keepdim=True)  # 归一化
    patch_embeddings = patch_embeddings[:, 1:, :]  # 去掉第一个 CLS token

    pil_img = Image.open(image_path)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # 1x4096x256 1x256
    similarity = compute_similarity_map(patch_embeddings[0:1, ...], text_features)
    similarity_map = get_similarity_map(similarity, cv2_img.shape[:2])
    vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
    vis = 255 - vis
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = cv2_img * 0.3 + vis * 0.7
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    plt.subplot(1, 1, 1) 
    plt.imshow(vis, cmap='viridis')
    plt.title('similarity_sam_map')
    plt.show()

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

    import torch.nn.functional as F
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