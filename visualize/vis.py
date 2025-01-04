# for debug use
def compute_similarity_map(
    image_features, 
    text_features, 
    redundant_feats=None
):  
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

# sm shape N_t
def similarity_map_to_points(sm, shape, t=0.2, down_sample=2):
#     (Pdb) p sm.shape
# torch.Size([1024])
    side = int(sm.shape[0] ** 0.5)
# (Pdb) p sm.shape
# torch.Size([1, 1, 32, 32])
    sm = sm.reshape(1, 1, side, side)

    # down sample to smooth results
    down_side = side // down_sample
    # (Pdb) sm.shape
    # torch.Size([16, 16])
    sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
    h, w = sm.shape
    sm = sm.reshape(-1)

    sm = (sm - sm.min()) / (sm.max() - sm.min())
    rank = sm.sort(0)[1]
    scale_h = float(shape[0]) / h
    scale_w = float(shape[1]) / w

    num = min((sm >= t).sum(), sm.shape[0] // 2)
    labels = np.ones(num * 2).astype('uint8')
    labels[num:] = 0
    points = []

    # positives
    for idx in rank[-num:]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1) # +0.5 to center
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    # negatives
    for idx in rank[:num]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1)
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])
    return points, labels

import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 加载模型和处理器
model = CLIPModel.from_pretrained("../../dataset_sesame/clip-vit-large-patch14-336").cuda()
processor = CLIPProcessor.from_pretrained("../../dataset_sesame/clip-vit-large-patch14-336")

# 输入文本和图像
text = "antler"
# text = "stool"
# text  = 'straw'
# text = 'person'
image_path = "../../dataset_sesame/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg"
# image_path = '../../dataset_sesame/reason_seg/ReasonSeg/val/scene0104_00_0.jpg'
# image_path = '../../dataset_sesame/reason_seg/ReasonSeg/val/2881277421_416273151c_o.jpg'
# image_path = 'assets/demo.jpg'
image = Image.open(image_path)

# 处理输入
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    outputs = model.vision_model(inputs['pixel_values'], output_hidden_states=True)
    patch_embeddings = outputs.hidden_states[-1]  # 最后一层的 patch 特征
    text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Patch embedding: 去掉 CLS token，剩下的就是 patch embedding
# patch_embeddings = patch_embeddings[:, 1:, :]  # 去掉第一个 CLS token
patch_embeddings = patch_embeddings[:, :, :]  # 去掉第一个 CLS token
patch_embeddings = model.visual_projection(patch_embeddings)

# 计算文本特征和 patch 特征的相似度
text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
patch_embeddings = patch_embeddings / patch_embeddings.norm(dim=-1, keepdim=True)  # 归一化
patch_embeddings = patch_embeddings[:, 1:, :]  # 去掉第一个 CLS token

debug = True
if debug:
    from PIL import Image
    import cv2
    from  matplotlib import pyplot as plt
    import numpy as np
    # image_path = "../../dataset_sesame/reason_seg/ReasonSeg/val/scene0104_00_0.jpg" #chair
    # image_path = "../../dataset_sesame/reason_seg/ReasonSeg/val/206674234_4cb520b13d_o.jpg"
    # image_path = "../../dataset_sesame/reason_seg/ReasonSeg/train/4971309080_370ab0baf3_o.jpg" #deer
    # image_path = "../../dataset_sesame/reason_seg/ReasonSeg/val/2881277421_416273151c_o.jpg"
    # image_path = "../../dataset_sesame/reason_seg/ReasonSeg/val/536167533_22228a08df_o.jpg"
    pil_img = Image.open(image_path)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # 1x4096x256 1x256    
    similarity = compute_similarity_map(patch_embeddings[0:1, ...], text_features)
    similarity_map = get_similarity_map(similarity, cv2_img.shape[:2])
    vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')

    points, labels = similarity_map_to_points(similarity[0, :, 0], cv2_img.shape[:2], t=0.8)
    for i, [x, y] in enumerate(points):
            # cv2.circle(vis, (x, y), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
            if labels[i] == 1:
               cv2.circle(vis, (x, y), 3, (0, 102, 255), 3)
               
    # vis = 255 - vis
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = cv2_img * 0.3 + vis * 0.7
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    plt.subplot(1, 1, 1) 
    plt.imshow(vis, cmap='viridis')
    plt.title('similarity_sam_map')
    plt.show()

debug = False
if debug:
    import clip
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    from  matplotlib import pyplot as plt
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    from segment_anything import sam_model_registry, SamPredictor
    
    ### Init CLIP and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_img = Image.open("assets/4971309080_370ab0baf3_o.jpg") # deer
    # pil_img = Image.open("scene0104_00_0.jpg") # chair
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ### Explain CLIP via our CLIP Surgery
    model, preprocess = clip.load("../dataset_sesame/CS-ViT-L-14-336px.pt", device=device)
    model.eval()

    # # This preprocess for all next cases
    preprocess =  Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

    ### CLIP Surgery for a single text, without fixed label sets
    # texts = ['stool']
    texts = [ 'antlers' ]

    with torch.no_grad():
        # CLIP architecture surgery acts on the image encoder
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)
        import pdb;pdb.set_trace()

        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)

        # Apply feature surgery for single text
        similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

        # Draw similarity map
        for b in range(similarity_map.shape[0]):
            for n in range(similarity_map.shape[-1]):
                vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = cv2_img * 0.5 + vis * 0.5
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                print('CLIP Surgery for a single text:', texts[n])
                plt.imshow(vis)
                plt.show()

