import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_small = [
    'a painting in {} view',
    'a rendering in {} view',
    'a cropped painting in {} view',
    'the painting in {} view',
    'a clean painting in {} view',
    'a dirty painting in {} view',
    'a dark painting in {} view',
    'a picture in {} view',
    'a cool painting in {} view',
    'a close-up painting in {} view',
    'a bright painting in {} view',
    'a cropped painting in {} view',
    'a good painting in {} view',
    'a close-up painting in {} view',
    'a rendition in {} view',
    'a nice painting in {} view',
    'a small painting in {} view',
    'a weird painting in {} view',
    'a large painting in {} view',
]

imagenet_dual_templates_small = [
    'a painting in {} view with {}',
    'a rendering in {} view with {}',
    'a cropped painting in {} view with {}',
    'the painting in {} view with {}',
    'a clean painting in {} view with {}',
    'a dirty painting in {} view with {}',
    'a dark painting in {} view with {}',
    'a cool painting in {} view with {}',
    'a close-up painting in {} view with {}',
    'a bright painting in {} view with {}',
    'a cropped painting in {} view with {}',
    'a good painting in {} view with {}',
    'a painting of one {} in {} view',
    'a nice painting in {} view with {}',
    'a small painting in {} view with {}',
    'a weird painting in {} view with {}',
    'a large painting in {} view with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(self.placeholder_token)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example