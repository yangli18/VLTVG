import numpy as np
import random
import math
import re

import torch
import PIL
from PIL import ImageFilter, ImageEnhance

import torchvision.transforms as T
import torchvision.transforms.functional as F
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, img, target):
        if random.random() < 0.8:
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    img = self.rand_brightness(img)
                elif func_id == 1:
                    img = self.rand_contrast(img)
                elif func_id == 2:
                    img = self.rand_saturation(img)

        return img, target


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img

        return input_dict




class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            w, h = img.size
            target['bbox'] = target['bbox'][[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            target['phrase'] = target['phrase'].replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
        return img, target


class ToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, img, target):
        for k in self.keys:
            target[k] = torch.as_tensor(target[k], dtype=torch.float32)
        return F.to_tensor(img), target


class RandomResize(object):
    def __init__(self, sizes, resize_long_side=True, record_resize_info=False):
        self.sizes = sizes
        self.resize_long_side = resize_long_side
        if resize_long_side:
            self.choose_size = max
        else:
            self.choose_size = min
        self.record_resize_info = record_resize_info

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        h, w = img.height, img.width
        ratio = float(size) / self.choose_size(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        img = F.resize(img, (new_h, new_w))

        ratio_h, ratio_w = float(new_h) / h, float(new_w) / w
        target['bbox'] = target['bbox'] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        if self.record_resize_info:
            target['orig_size'] = torch.as_tensor([h, w], dtype=torch.float32)
            target['ratio'] = torch.as_tensor([ratio_h, ratio_w], dtype=torch.float32)
            target['size'] = torch.as_tensor([new_h, new_w], dtype=torch.float32)

        return img, target


def crop(image, box, region):
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])
    cropped_box = torch.min(cropped_box.reshape(-1, 2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)
    # area = (cropped_box[:, 1, :] - cropped_box[:, 0, :]).prod(dim=1)

    return cropped_image, cropped_box


class RandomSizeCrop(object):
    def __init__(self, min_size, max_size, max_cnt=20, check_method={}):
        self.min_size = min_size
        self.max_size = max_size
        self.max_cnt = max_cnt

        if check_method.get('func', 'area') == 'area':
            self.check = self.check_area
            self.area_thres = check_method.get('area_thres', 0)
        elif check_method.get('func') == 'iou':
            self.check = self.check_iou
            self.iou_thres = check_method.get('iou_thres', 0.5)
        else: raise NotImplementedError

    def __call__(self, img, target):
        for i in range(self.max_cnt):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            i, j, th, tw = region
            orig_box = target['bbox']
            cropped_box = orig_box - torch.as_tensor([j, i, j, i])
            cropped_box = torch.min(cropped_box.reshape(-1, 2, 2), torch.as_tensor([tw, th], dtype=torch.float32))
            cropped_box = cropped_box.clamp(min=0).reshape(-1) + torch.as_tensor([j, i, j, i])
            if self.check(cropped_box, orig_box):
                img, box = crop(img, orig_box, region)
                target['bbox'] = box
                return img, target

        return img, target

    def check_iou(self, cropped_box, orig_box):
        iou = box_iou(cropped_box.view(-1, 4), orig_box.view(-1, 4))[0]
        return (iou >= self.iou_thres).all()

    def check_area(self, cropped_box, orig_box):
        cropped_box = cropped_box.reshape(-1, 2, 2)
        box_hw = cropped_box[:, 1, :] - cropped_box[:, 0, :]
        return (box_hw > 0).all() and (box_hw.prod(dim=1) > self.area_thres).all()


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5,
                 exclude_words=['left', 'right', 'top', 'bottom', 'middle']):
        args = transforms1.copy()
        self.transforms1 = PIL_TRANSFORMS[args.pop('type')](**args)
        args = transforms2.copy()
        self.transforms2 = PIL_TRANSFORMS[args.pop('type')](**args)
        self.p = p
        self.exclude_words = exclude_words

    def __call__(self, img, target):
        phrase = target['phrase']
        for word in self.exclude_words:
            if word in phrase:
                return self.transforms1(img, target)

        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = list()
        for t in transforms:
            args = t.copy()
            transform = PIL_TRANSFORMS[args.pop('type')](**args)
            self.transforms.append(transform)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class NormalizeAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640,
                 aug_translate=False, center_place=False, padding=True):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate
        self.center_place = center_place
        self.padding = padding

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        elif self.center_place:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)
        else:
            top = left = 0

        if self.padding:
            out_img = torch.zeros((3, self.size, self.size)).float()
            out_img[:, top:top + h, left:left + w] = img
            out_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
            target['mask'] = out_mask
        else:
            out_img = img

        box = target['bbox']
        box[0], box[2] = box[0] + left, box[2] + left
        box[1], box[3] = box[1] + top, box[3] + top
        target['dxdy'] = torch.as_tensor([left, top], dtype=torch.float32)

        out_h, out_w = out_img.shape[-2:]
        box = box_xyxy_to_cxcywh(box)
        box = box / torch.tensor([out_w, out_h, out_w, out_h], dtype=torch.float32)
        target['size'] = torch.as_tensor([out_h, out_w], dtype=torch.float32)
        target['bbox'] = box

        return out_img, target


PIL_TRANSFORMS = {
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'ToTensor': ToTensor,
    'RandomResize': RandomResize,
    'RandomSizeCrop': RandomSizeCrop,
    'NormalizeAndPad': NormalizeAndPad,
    'RandomSelect': RandomSelect,
    'Compose': Compose,
    'ColorJitter': ColorJitter,
    'GaussianBlur': GaussianBlur,
}