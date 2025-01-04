import cv2
import torch
import numpy as np

import requests
from io import BytesIO
import torch.nn.functional as F
from model.segment_anything.utils.transforms import ResizeLongestSide


def load_image(path_or_url):
    if path_or_url.startswith('http'):  # Checks if the path is a URL
        response = requests.get(path_or_url)  # Fetch the image via HTTP
        image_bytes = BytesIO(response.content)  # Convert to a Bytes stream
        image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode the image
    else:
        image = cv2.imread(path_or_url, cv2.IMREAD_COLOR)  # Load image from file path
    
    return image


class BaseDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_size = 1024
    ignore_label = 255

    def __init__(
        self,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        return image, image_clip, sam_input_shape

    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # You should implement this method yourself!
        return NotImplementedError
    

class ImageProcessor:
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    image_size = 1024
    ignore_label = 255

    def __init__(
        self,
        vision_tower,
        image_size: int = 336,
    ):
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_and_preprocess_image(self, image_path):
        image = load_image(image_path)
        sam_output_shape = tuple(image.shape[:2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        sam_input_shape = tuple(image.shape[:2])
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        sam_mask_shape = [sam_input_shape, sam_output_shape]
        return image, image_clip, sam_mask_shape