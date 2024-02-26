from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
import options
from custom_types import *
from utils import files_utils



@dataclass
class Transform:
    offset_h: int
    offset_w: int
    h: int
    w: int
    is_flip: bool


def resize_image(image: ARRAY, region: options.Region, is_train: bool) -> ARRAY:
    h, w, c = image.shape
    if is_train:
        target_w = int(region.width / region.crop_ratio_min)
        target_h = int(region.height / region.crop_ratio_min)
    else:
        target_w = region.width
        target_h = region.height
    ar_source = float(h) / w
    ar_target = float(target_h) / target_w
    if ar_source < ar_target:
        target_w = int(target_h / ar_source)
    else:
        target_h = int(target_w * ar_source)
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h), Image.BICUBIC)
    return V(image)


def sample_transform(source_shape, region: options.Region, is_train: bool) -> Transform:
    h, w = source_shape
    if is_train:
        crop_ratio = region.crop_ratio_min + float(np.random.rand(1)) * (1 - region.crop_ratio_min)
        target_h, target_w = int(h * crop_ratio), int(w * crop_ratio)
        return Transform(int(np.random.randint(0, h - target_h + 1, (1,))),
                         int(np.random.randint(0, w - target_w + 1, (1,))),
                         target_h,
                         target_w,
                         region.random_flip and int(np.random.randint(0, 2, (1,))) == 1)
    else:
        return Transform(0, 0, h, w, False)


def apply_transform(image: ARRAY, transform: Transform, region: options.Region) -> ARRAY:
    image = image[transform.offset_h: transform.offset_h + transform.h,
            transform.offset_w: transform.offset_w + transform.w]
    if transform.is_flip:
        image = image[:, ::-1, :]
    h, w, c = image.shape
    if h > w:
        offset = (h - w) // 2
        image = image[offset: offset + w]
    elif w > h:
        offset = (w - h) // 2
        image = image[:, offset: offset +h]
    h, w, c = image.shape
    if h != region.height:
        image = Image.fromarray(image).resize((region.width, region.height), Image.BICUBIC)
    return V(image)


def load_and_resize_image(image_path) -> T:
    image = files_utils.load_image(image_path)
    h, w, c = image.shape
    target_w = target_h = 512
    ar_source = float(h) / w
    ar_target = 1.
    if ar_source < ar_target:
        target_w = int(target_h / ar_source)
    else:
        target_h = int(target_w * ar_source)
    image = Image.fromarray(image)
    image = V(image.resize((target_w, target_h), Image.BICUBIC))
    h, w, _ = image.shape
    offset_h_, offset_w_ = (h - target_h) // 2, (w - target_w) // 2
    image = image[offset_h_: offset_h_ + target_h, offset_w_: offset_w_ + target_w]
    return to_pt(image)


def to_pt(image, to_minus_plus_one=True):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.
    if to_minus_plus_one:
        image = image * 2 - 1
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    return image


class ImageDS(Dataset):

    def transform(self, image: ARRAY) -> T:
        h, w, c = image.shape
        transform = sample_transform((h, w,), self.args.region, self.is_train)
        image = apply_transform(image, transform, self.args.region)
        return to_pt(image)

    def __getitem__(self, item) -> tuple[T, str]:
        item = item % len(self.images)
        out = self.transform(self.images[item])
        prompt = self.args.text_prompt
        if not isinstance(prompt, str):
            prompt = prompt[int(np.random.randint(0, len(prompt), 1))]
        return out, prompt

    def __len__(self) -> int:
        return 1_000_000

    def __init__(self, args: options.ArgsAdapter, is_train: bool):
        self.args = args
        self.is_train = is_train

        paths = [args.image_path] if isinstance(args.image_path, str)  else args.image_path
        images = [files_utils.load_image(path) for path in paths]
        self.images = [resize_image(image, args.region, is_train) for image in images]
