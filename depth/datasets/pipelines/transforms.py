import cv2
import mmcv
import torch
import numpy as np
from ..builder import PIPELINES
from numpy.core.fromnumeric import shape
from mmcv.utils import deprecated_api_warning

from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import AutoAugment, RandAugment, rand_augment_transform
import os.path as path
# bash tools/dist_train7.sh
import random
from imgaug import augmenters as iaa


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean,
                                       std=self.std,
                                       to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class NYUCrop(object):
    """NYU standard krop when training monocular depth estimation on NYU dataset.

    Args:
        depth (bool): Whether apply NYUCrop on depth map. Default: False.
    """
    def __init__(self, depth=False):
        self.depth = depth

    def __call__(self, results):
        """Call function to apply NYUCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """

        if results['img_info'].get('dataset') is None or results['img_info'].get('dataset') == 'NYU_RGBD':
            for key in results.get('depth_fields', []):
                valid_mask = np.zeros_like(results[key])
                valid_mask[45:472, 43:608] = 1
                results[key][valid_mask == 0] = 0

                # results[key] = results[key][45:472, 43:608]

            # img_cropped = results["img"][45:472, 43:608, :]
            # results["img"] = img_cropped
            # results["ori_shape"] = img_cropped.shape

        # if self.depth:
        #     depth_cropped = results["depth_gt"][45:472, 43:608]
        #     results["depth_gt"] = depth_cropped
        #     results["depth_shape"] = results["depth_gt"].shape
        #
        # img_cropped = results["img"][45:472, 43:608, :]
        # results["img"] = img_cropped
        # results["ori_shape"] = img_cropped.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class NYUMask(object):
    # def __init__(self):
    def __call__(self, results):
        depth_gt = results["depth_gt"]
        valid_mask = np.zeros_like(depth_gt)
        valid_mask[45:472, 43:608] = 1
        depth_gt[valid_mask == 0] = 0
        results["depth_gt"] = depth_gt
        # translation_matrix = np.float32([[1, 0, -45 + np.random.rand() * 53],
        #                                  [0, 1, -43 + np.random.rand() * 75]])
        #
        # results["depth_gt"] = cv2.warpAffine(results["depth_gt"], translation_matrix, (640, 480))
        # results["img"] = cv2.warpAffine(results["img"], translation_matrix, (640, 480))

        # img = results['img']
        # depth_gt = results["depth_gt"]
        # valid_mask = np.zeros_like(depth_gt)
        # valid_mask[45:472, 43:608] = 1
        # depth_gt[valid_mask == 0] = 0
        # # img[valid_mask == 0] = 0
        #
        # if np.random.rand() < 0.25:
        #     translation_matrix = np.float32([[1, 0, -45 + np.random.rand() * 53],
        #                                     [0, 1, -43 + np.random.rand() * 75]])
        #     img = cv2.warpAffine(img, translation_matrix, (640, 480))
        #     depth_gt = cv2.warpAffine(depth_gt, translation_matrix, (640, 480))
        #
        # results["img"] = img
        # results["depth_gt"] = depth_gt


        # if np.random.rand() < 0.5:
        #     # depth_gt = results["depth_gt"]
        #     # img = results['img']
        #     depth_gt = results["depth_gt"]
        #     valid_mask = np.zeros_like(depth_gt)
        #     valid_mask[45:472, 43:608] = 1
        #     depth_gt[valid_mask == 0] = 0
        #     # img[valid_mask == 0] = 0
        #     # results["img"] = img
        #     results["depth_gt"] = depth_gt
        # else:
        #     img = results['img'][45:472, 43:608, :]
        #     depth_gt = results["depth_gt"][45:472, 43:608]
        #     results['img'] = mmcv.imresize(img, (640, 480),)
        #     results["depth_gt"] = mmcv.imresize(depth_gt, (640, 480),)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class SUN_Resize(object):
    def __init__(self, size=(640, 480)):
        self.size = size

    def __call__(self, results):
        img = results['img']
        results['img'] = mmcv.imresize(img, size=self.size, interpolation='bilinear')
        results["ori_shape"] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class DIODE_Resize(object):
    def __init__(self, depth=False, size=(640, 480)):
        self.depth = depth
        self.size = size

    def __call__(self, results):
        img = results['img']
        results['img'] = mmcv.imresize(img, size=self.size, interpolation='bilinear')
        results["ori_shape"] = results['img'].shape
        if self.depth:
            depth = results["depth_gt"]
            results["depth_gt"] = mmcv.imresize(depth, size=self.size, interpolation='bilinear')
            results["depth_shape"] = results["depth_gt"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class KBCrop(object):
    """KB standard krop when training monocular depth estimation on KITTI dataset.

    Args:
        depth (bool): Whether apply KBCrop on depth map. Default: False.
        height (int): Height of input images. Default: 352.
        width (int): Width of input images. Default: 1216.

    """
    def __init__(self, depth=False, height=352, width=1216):
        self.depth = depth
        self.height = height
        self.width = width

    def __call__(self, results):
        """Call function to apply KBCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """
        # print(results)
        height = results["img_shape"][0]
        width = results["img_shape"][1]
        top_margin = int(height - self.height)
        left_margin = int((width - self.width) / 2)

        if self.depth:
            depth_cropped = results["depth_gt"][top_margin:top_margin +
                                                self.height,
                                                left_margin:left_margin +
                                                self.width]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape

        img_cropped = results["img"][top_margin:top_margin + self.height,
                                     left_margin:left_margin + self.width, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class Padding(object):
    def __init__(self, img_padding_value, depth_padding_value, normals=False, pe_k=False, ori_h=352, ori_w=1216):
        self.img_padding_value = img_padding_value
        self.depth_padding_value = depth_padding_value
        self.normals = normals
        self.pe_k = pe_k
        self.ori_h = ori_h
        self.ori_w = ori_w

    def pad_img(self, image):
        image_dtype = image.dtype
        img_h, img_w, channel_ = image.shape

        if img_h < self.ori_h or img_w < self.ori_w:
            pad_h, pad_w = img_h, img_w
            if self.ori_h - img_h > 0:
                h_off = random.randint(0, self.ori_h - img_h)
                pad_h = self.ori_h
            else:
                h_off = 0
            if self.ori_w - img_w > 0:
                w_off = random.randint(0, self.ori_w - img_w)
                pad_w = self.ori_w
            else:
                w_off = 0

            new_img = np.zeros((pad_h, pad_w, channel_)).astype(image_dtype)

            new_img[h_off:h_off + img_h, w_off:w_off + img_w] = image
            return new_img
        else:
            return image
    def pad_depth(self, image):
        image_dtype = image.dtype
        img_h, img_w = image.shape

        if img_h < self.ori_h or img_w < self.ori_w:
            pad_h, pad_w = img_h, img_w
            if self.ori_h - img_h > 0:
                h_off = random.randint(0, self.ori_h - img_h)
                pad_h = self.ori_h
            else:
                h_off = 0
            if self.ori_w - img_w > 0:
                w_off = random.randint(0, self.ori_w - img_w)
                pad_w = self.ori_w
            else:
                w_off = 0

            new_img = np.zeros((pad_h, pad_w)).astype(image_dtype)

            new_img[h_off:h_off + img_h, w_off:w_off + img_w] = image
            return new_img
        else:
            return image

    def __call__(self, results):
        results['img'] = self.pad_img(results['img'])
        for key in results.get('depth_fields', []):
            results[key] = self.pad_depth(results[key])
        # image = results['img'].copy()
        # depth = results['depth_gt'].copy()

        return results


@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & depth.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        depth_pad_val (float, optional): Padding value of depth map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """
    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 depth_pad_val=0,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.depth_pad_val = depth_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(results['img'],
                                           angle=degree,
                                           border_value=self.pal_val,
                                           center=self.center,
                                           auto_bound=self.auto_bound)

            # rotate depth
            for key in results.get('depth_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.depth_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'depth_pad_val={self.depth_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & depth.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, depth estimation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'],
                                         direction=results['flip_direction'])

            # flip depth
            for key in results.get('depth_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """
    def __init__(self, crop_size, resize_prob=0):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.rp = resize_prob

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def get_pad_box(self, image):
        img_h, img_w, = image.shape[:2]
        h, w = self.crop_size
        if img_h < h or img_w < w:
            pad_h, pad_w = img_h, img_w
            if h - img_h > 0:
                h_off = random.randint(0, h - img_h)
                pad_h = h
            else:
                h_off = 0
            if w - img_w > 0:
                w_off = random.randint(0, w - img_w)
                pad_w = w
            else:
                w_off = 0

            return h_off, img_h, w_off, img_w, pad_h, pad_w
        else:
            return None

    def pad_img(self, image, box, pad_value=0):
        if box is None:
            return image

        image_dtype = image.dtype
        h_off, img_h, w_off, img_w, pad_h, pad_w = box

        if image.ndim == 3:
            new_img = np.zeros((pad_h, pad_w, image.shape[-1])).astype(image_dtype) + pad_value
        else:
            new_img = np.zeros((pad_h, pad_w)).astype(image_dtype) + pad_value

        new_img[h_off:h_off + img_h, w_off:w_off + img_w] = image
        return new_img


    def crop(self, img, crop_bbox, resize, pad_box, pad_value=0):
        """Crop from ``img``"""

        if resize:
            h, w = self.crop_size
            img = mmcv.imresize(img, size=(w, h), interpolation='nearest')
            return img

        img = self.pad_img(img, pad_box, pad_value)
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        pad_box = self.get_pad_box(img)
        # if self.rp > 0:
        resize = np.random.rand() < self.rp

        depth = self.crop(results["depth_gt"], crop_bbox, resize, pad_box)
        # valid = (np.mean(depth > 0) >= 1 / 3.) and (np.std(depth[depth > 0]) > 0.2 if np.any(depth > 0) else False)
        valid = np.mean(depth > 1e-3) >= 1 / 4.
        if valid:
            # print(results['img_info']['dataset'])
            results["depth_gt"] = depth
        else:
            # print('resize!', np.mean(depth > 0), np.std(depth[depth > 0]))
            # print(results['img_info']['dataset'])
            resize = True
            results["depth_gt"] = self.crop(results["depth_gt"], crop_bbox, resize, pad_box)

        # crop the image
        img = self.crop(img, crop_bbox, resize, pad_box)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop depth


        if 'edge_map' in results.get('depth_fields', []):
            results['edge_map'] = self.crop(results['edge_map'], crop_bbox, resize, pad_box)

        if results.get('mix_img') is not None:
            results['mix_img'] = self.crop(results['mix_img'], crop_bbox, resize, pad_box)
            results["mix_depth"] = self.crop(results["mix_depth"], crop_bbox, resize, pad_box)
            results["mix_edge"] = self.crop(results["mix_edge"], crop_bbox, resize, pad_box)

        results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomCropPad(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size, crop_range=(352, 416)):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.crop_range = crop_range

    def get_crop_size(self, img):
        # """Randomly get a crop bounding box."""
        # h_max = min(img.shape[0], self.crop_range[0])
        # w_max = min(img.shape[1], self.crop_range[1])
        # h = np.random.randint(self.crop_range[0], h_max+1)
        # w = np.random.randint(self.crop_range[1], w_max+1)
        #
        h_max = img.shape[0]
        w_max = img.shape[1]
        if h_max > self.crop_range[0]:
            h = np.random.randint(self.crop_range[0], h_max + 1)
        else:
            h = h_max

        if h_max > self.crop_range[1]:
            w = np.random.randint(self.crop_range[1], w_max + 1)
        else:
            w = w_max

        self.crop_size = (h, w)
        return h, w

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        img = results['img']
        h, w = self.get_crop_size(img)
        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop depth
        for key in results.get('depth_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class CenterCrop(object):
    """Center crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """
    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Get a center crop bounding box."""
        h, w = img.shape[0], img.shape[1]
        ch, cw = self.crop_size

        start_h = max((h - ch) // 2, 0)
        start_w = max((w - cw) // 2, 0)
        end_h = start_h + ch
        end_w = start_w + cw

        return start_h, end_h, start_w, end_w

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to center crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Center cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop depth
        for key in results.get('depth_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomMask(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size, prob=.5):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.prob = prob

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        # crop_size_h = np.random.randint(self.crop_size[0], img.shape[0])
        crop_size_w = np.random.randint(self.crop_size[1], img.shape[1])
        # margin_h = max(img.shape[0] - self.crop_size[0], 0)
        # margin_w = max(img.shape[1] - self.crop_size[1], 0)
        # margin_h = max(img.shape[0] - crop_size_h, 0)
        margin_w = max(img.shape[1] - crop_size_w, 0)
        # offset_h = np.random.randint(0, margin_h + 1)
        offset_h = 0
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, depth=False):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        valid_mask = np.zeros_like(img)
        valid_mask[crop_y1:crop_y2, crop_x1:crop_x2, ...] = 1
        # img[valid_mask == 0] = 0
        if depth or np.random.random(1) > 0.5:
            img[valid_mask == 0] = 0
        else:
            img[valid_mask == 0] = 255
        return img

    def __call__(self, results):
        """Call function to randomly crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        mask = True if np.random.rand() < self.prob else False
        if mask:
            img = results['img']
            crop_bbox = self.get_crop_bbox(img)

            # crop the image
            img = self.crop(img, crop_bbox)
            # img_shape = img.shape
            results['img'] = img
            # results['img_shape'] = img_shape

            # crop depth
            for key in results.get('depth_fields', []):
                results[key] = self.crop(results[key], crop_bbox, depth=True)

            # results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class CutDepth(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, prob=.5):
        # assert crop_size[0] > 0 and crop_size[1] > 0
        # self.crop_size = crop_size
        self.prob = prob

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        crop_size_y = np.random.randint(1, img.shape[0])
        crop_size_x = np.random.randint(1, img.shape[1])
        crop_size_h = np.random.randint(crop_size_y, img.shape[0])
        crop_size_w = np.random.randint(crop_size_x, img.shape[1])
        # margin_h = max(img.shape[0] - self.crop_size[0], 0)
        # margin_w = max(img.shape[1] - self.crop_size[1], 0)
        margin_h = max(img.shape[0] - crop_size_h, 0)
        margin_w = max(img.shape[1] - crop_size_w, 0)
        # offset_h = np.random.randint(0, margin_h + 1)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size_y
        crop_x1, crop_x2 = offset_w, offset_w + crop_size_x

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, depth=False):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        # img[valid_mask == 0] = 0
        if depth is not None and np.random.rand() < self.prob:
            # depth = abs(depth)
            img[crop_y1:crop_y2, crop_x1:crop_x2, 0] = depth[crop_y1:crop_y2, crop_x1:crop_x2]
            img[crop_y1:crop_y2, crop_x1:crop_x2, 1] = depth[crop_y1:crop_y2, crop_x1:crop_x2]
            img[crop_y1:crop_y2, crop_x1:crop_x2, 2] = depth[crop_y1:crop_y2, crop_x1:crop_x2]
        # elif np.random.rand() < self.prob:
        #     valid_mask = np.zeros_like(img)
        #     valid_mask[crop_y1:crop_y2, crop_x1:crop_x2, ...] = 1
        #     img[valid_mask == 0] = 255
        return img

    def __call__(self, results):
        """Call function to randomly crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        mask = True if np.random.rand() < self.prob else False
        if mask:
            img = results['img']
            crop_bbox = self.get_crop_bbox(img)

            # crop the image
            img = self.crop(img, crop_bbox, results["depth_gt"])
            # img_shape = img.shape
            results['img'] = img
            # results['img_shape'] = img_shape

            # crop depth
            # for key in results.get('depth_fields', []):
            #     results[key] = self.crop(results[key], crop_bbox, depth=True)

            # results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__




@PIPELINES.register_module()
class CutMix(object):
    # def __init__(self):
    #     self.prob = prob

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        crop_size_y = np.random.randint(1, img.shape[0])
        crop_size_x = np.random.randint(1, img.shape[1])
        crop_size_h = np.random.randint(crop_size_y, img.shape[0])
        crop_size_w = np.random.randint(crop_size_x, img.shape[1])
        margin_h = max(img.shape[0] - crop_size_h, 0)
        margin_w = max(img.shape[1] - crop_size_w, 0)
        # offset_h = np.random.randint(0, margin_h + 1)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size_y
        crop_x1, crop_x2 = offset_w, offset_w + crop_size_x

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, img2):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img[crop_y1:crop_y2, crop_x1:crop_x2, ...] = img2[crop_y1:crop_y2, crop_x1:crop_x2, ...]

        return img

    def __call__(self, results):

        if results.get('mix_img') is not None:
            img = results['img']
            crop_bbox = self.get_crop_bbox(img)
            img2 = results['mix_img']

            depth = results['depth_gt']
            depth2 = results['mix_depth']

            edge = results['edge_map']
            edge2 = results['mix_edge']


            results['img'] = self.crop(img, crop_bbox, img2)
            results['depth_gt'] = self.crop(depth, crop_bbox, depth2)
            results['edge_map'] = self.crop(edge, crop_bbox, edge2)

            # del results['mix_img']
            # del results['mix_depth']
            # del results['mix_edge']

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Bicubic(object):
    def __init__(self, hw=(616, 1064)):
        # self.h, self.w = hw
        self.max_area = hw[0] * hw[1]

    def resize(self, img):
        h, w = img.shape[:2]
        # long_side = max(h, w)
        # print(h, w)
        cur_area = h * w
        if cur_area > self.max_area:
            # scale = 1024 / long_side
            scale = (self.max_area / cur_area)**0.5
            img, scale_factor = mmcv.imrescale(img,
                                               scale,
                                               interpolation='nearest',
                                               return_scale=True)

        # img = mmcv.imresize(img, size=(self.w, self.h), interpolation='nearest')

        return img

    def __call__(self, results):
        results['img'] = self.resize(results['img'])
        for key in results.get('depth_fields', []):
            results[key] = self.resize(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ColorAug(object):
    """Color augmentation used in depth estimation

    Args:
        prob (float, optional): The color augmentation probability. Default: None.
        gamma_range(list[int], optional): Gammar range for augmentation. Default: [0.9, 1.1].
        brightness_range(list[int], optional): Brightness range for augmentation. Default: [0.9, 1.1].
        color_range(list[int], optional): Color range for augmentation. Default: [0.9, 1.1].
    """
    def __init__(self,
                 prob=None,
                 gamma_range=[0.9, 1.1],
                 brightness_range=[0.9, 1.1],
                 color_range=[0.9, 1.1]):
        self.prob = prob
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to apply color augmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly colored results.
        """
        aug = True if np.random.rand() < self.prob else False

        if aug:
            image = results['img']

            # gamma augmentation
            gamma = np.random.uniform(min(*self.gamma_range),
                                      max(*self.gamma_range))
            image_aug = image**gamma

            # brightness augmentation
            brightness = np.random.uniform(min(*self.brightness_range),
                                           max(*self.brightness_range))
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(min(*self.color_range),
                                       max(*self.color_range),
                                       size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)],
                                   axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 255)

            results['img'] = image_aug

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Resize(object):
    """Resize images & depth.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 normals=False,
                 prob=0.25,
                 ):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.normals = normals
        self.prob = prob

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long),
                                      max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short),
                                       max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range, keep_ratio=True):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        if keep_ratio:
            ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        else:
            ratio1 = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            ratio2 = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            scale = int(img_scale[0] * ratio1), int(img_scale[1] * ratio2)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range,
                                                            self.keep_ratio
                                                            )
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        interp = random.choice(['nearest', 'bilinear', 'bicubic'])
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               interpolation=interp,
                                               return_scale=True)

            if results.get('mix_img') is not None:
                results['mix_img'] = mmcv.imrescale(results['mix_img'],
                                               results['scale'],
                                               interpolation=interp)


            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  interpolation=interp,
                                                  return_scale=True)

            if results.get('mix_img') is not None:
                results['mix_img'] = mmcv.imresize(results['mix_img'],
                                                   results['scale'],
                                                   interpolation=interp)

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_depth(self, results):
        """Resize depth estimation map with ``results['scale']``."""
        for key in results.get('depth_fields', []):
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(results[key],
                                          results['scale'],
                                          interpolation='nearest'
                                          )
            else:
                gt_depth = mmcv.imresize(results[key],
                                         results['scale'],
                                         interpolation='nearest'
                                         )
            results[key] = gt_depth


    def __call__(self, results):
        do_it = random.random() <= self.prob
        if not do_it:
            return results

        if 'scale' not in results:
            self._random_scale(results)

        self._resize_img(results)
        self._resize_depth(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class RandomChannelSwap(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, prob):
        self.prob = prob
        from itertools import permutations
        self.indices = list(permutations(range(3), 3))

    def __call__(self, results):
        img = results['img']
        if np.random.rand() < self.prob:
            img = img[..., list(self.indices[np.random.randint(0, len(self.indices) - 1)])]

        results["img"] = img

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class GaussianBlur(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, prob=0.25):
        self.prob = prob
        # transforms.GaussianBlur
        # self.blur = gaussian_filter()
        self.blur = transforms.GaussianBlur(3)
        self.erase = RandomErasing(probability=0.25)
        self.radius_min = 0.1
        self.radius_max = 2.

    def __call__(self, results):
        img = results['img']
        # do_it = random.random() <= self.prob
        # if not do_it:
        #     return results

        # img = img.filter(
        #     ImageFilter.GaussianBlur(
        #         radius=random.uniform(self.radius_min, self.radius_max)
        #     )
        # )

        # img = self.erase(img)
        img = torch.from_numpy(img.copy())
        if np.random.rand() < self.prob:
            img = self.blur(img)
        # img = self.eraseI
        img = self.erase(img)
        results["img"] = np.array(img)

        return results




    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomEdgeMask(object):
    """
    Random mask the input and labels.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """
    def __init__(self, skip=['VKITTI_RGBD', 'Hypersim_RGBD']):
        self.datasets = skip

    def mask_edge(self, img, depth, canny_low: int = 100, canny_high: int = 200) -> np.ndarray:
        """
        img: H×W×3 的uint8或float图像（RGB或BGR都行）
        返回：边缘及其3×3邻域置零后的新图像
        """
        assert img.ndim == 3 and img.shape[2] == 3, "expect H×W×3 image"
        out = img.copy()

        # 灰度（OpenCV假定传入是BGR；若你的数据是RGB，先 out = out[..., ::-1] 再转）
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)  # 若是RGB：cv2.COLOR_RGB2GRAY

        # 1) Canny 边缘
        edges = cv2.Canny(gray, canny_low, canny_high)  # uint8 0/255

        # 2) 3×3 膨胀得到“边缘附近3×3区域”掩码
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1).astype(bool)  # True表示要置零的像素

        # 3) 对三通道同时置 0
        depth[mask] = 0

        return depth

    def __call__(self, results):
        if 'dataset' in results.get('img_info', {}):
            if results['img_info']['dataset'] not in self.datasets:
                img = results['img']
                results['depth_gt'] = self.mask_edge(img, results['depth_gt'])
            else:
                img = results['img']
                results['depth_gt'] = self.mask_edge(img, results['depth_gt'])
            # for key in results.get('depth_fields', []):
            #     gt_depth = results[key]
            #     results[key] = self.mask_edge(img, gt_depth)

        return results

@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 to_gray_prob=0.3,
                 distortion_prob=0.3,
                 **kwargs):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.gray_aug = iaa.Grayscale(alpha=(0.8, 1.0))
        self.to_gray_prob = to_gray_prob
        self.distortion_prob = distortion_prob

    def convert(self, img, alpha=1.0, beta=0.0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, beta, do):
        """Brightness distortion."""
        if do:
            # beta = random.uniform(-self.brightness_delta,
            #                         self.brightness_delta)
            img = self.convert(
                img,
                beta=beta)
        return img

    def contrast(self, img, alpha, do):
        """Contrast distortion."""
        if do:
            # alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(
                img,
                alpha=alpha)
        return img

    def saturation(self, img, alpha, do):
        """Saturation distortion."""
        if do:
            # alpha = random.uniform(self.saturation_lower,
            #                         self.saturation_upper)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=alpha)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img, rand_hue, do):
        """Hue distortion."""
        if do:
            # rand_hue = random.randint(-self.hue_delta, self.hue_delta)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def rgb2gray(self, img):
        img = self.gray_aug(image=img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        brightness_beta = random.uniform(-self.brightness_delta, self.brightness_delta)
        brightness_do = random.random() < self.distortion_prob

        contrast_alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        contrast_do = random.random() < self.distortion_prob

        saturate_alpha = random.uniform(self.saturation_lower, self.saturation_upper)
        saturate_do = random.random() < self.distortion_prob

        rand_hue = random.randint(-self.hue_delta, self.hue_delta)
        rand_hue_do = random.random() < self.distortion_prob

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = 1 if random.random() > 0.5 else 2
        img = results['img']
        if random.random() < self.to_gray_prob:
            img = self.rgb2gray(img)
        else:
            # random brightness
            img = self.brightness(img, brightness_beta, brightness_do)

            if mode == 1:
                img = self.contrast(img, contrast_alpha, contrast_do)

            # random saturation
            img = self.saturation(img, saturate_alpha, saturate_do)

            # random hue
            img = self.hue(img, rand_hue, rand_hue_do)

            # random contrast
            if mode == 0:
                img = self.contrast(img, contrast_alpha, contrast_do)

        results['img'] = img
        return results


@PIPELINES.register_module()
class Weather(object):
    """Apply the following weather augmentations to data.
    Args:
        prob (float): probability to enforce the weather augmentation.
    """

    def __init__(self,
                 prob=0.3,
                 **kwargs):
        snow = iaa.FastSnowyLandscape(
            lightness_threshold=[50, 100],
            lightness_multiplier=(1.2, 2)
        )
        cloud = iaa.Clouds()
        fog = iaa.Fog()
        snow_flakes = iaa.Snowflakes(flake_size=(0.2, 0.4),
                                     speed=(0.001, 0.03))  # iaa.imgcorruptlike.Snow(severity=2)#
        rain = iaa.Rain(speed=(0.1, 0.3), drop_size=(0.1, 0.3))
        # rain_drops = RainDrop_Augmentor()
        self.aug_list = [
            snow, cloud, fog, snow_flakes, rain,
            # wa.add_sun_flare, wa.darken, wa.random_brightness,
        ]
        self.prob = prob

    def aug_with_weather(self, imgs, id):
        weather = self.aug_list[id]
        return weather(image=imgs)

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if random.random() < self.prob:
            img = results['img']
            select_id = np.random.randint(0, high=len(self.aug_list))
            results['img'] = self.aug_with_weather(img, select_id)
        return results


@PIPELINES.register_module()
class ResizeCrop(object):
    def __init__(self,
                 crop_size,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 normals=False):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.normals = normals

        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long),
                                      max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short),
                                       max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)

        img_h, img_w, channel_ = img.shape
        if img_h < self.crop_size[0]:
            channel_ = img.shape[-1]
            new_img = np.zeros((self.crop_size[0], self.crop_size[1], channel_)).astype(img.dtype)
            # new_depth = np.zeros((self.ori_h, self.ori_w)).astype(depth_dtype)
            h_off = random.randint(0, self.crop_size[0] - img_h)
            w_off = random.randint(0, self.crop_size[1] - img_w)
            new_img[h_off:h_off + img_h, w_off:w_off + img_w] = img
            img = new_img
            # new_depth[h_off:h_off + img_h, w_off:w_off + img_w] = depth
        elif img_h > self.crop_size[0]:
            crop_box = self.get_crop_bbox(img)
            img = self.crop(img, crop_box)
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_box
            h_factor = self.crop_size[0] / img_h
            w_factor = self.crop_size[1] / img_w
            crop_y1, crop_y2, crop_x1, crop_x2 = (int(crop_y1*h_factor)+1, int(crop_y2*h_factor)-1,
                                                  int(crop_x1*w_factor)+1, int(crop_x2*w_factor)-1)
            for key in results.get('depth_fields', []):
                gt_depth = results[key]
                new_depth = np.zeros((self.crop_size[0], self.crop_size[1])).astype(gt_depth.dtype)
                new_depth[crop_y1:crop_y2, crop_x1:crop_x2] = 0
                results[key] = new_depth


            # [crop_y1: crop_y2, crop_x1: crop_x2, ...]
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_depth(self, results):
        """Resize depth estimation map with ``results['scale']``."""
        for key in results.get('depth_fields', []):
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(results[key],
                                          results['scale'],
                                          interpolation='nearest'
                                          )
            else:
                gt_depth = mmcv.imresize(results[key],
                                         results['scale'],
                                         interpolation='nearest'
                                         )
            results[key] = gt_depth


    def __call__(self, results):
        do_it = random.random() <= 0.5
        if not do_it:
            return results
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        # self._resize_depth(results)

        return results


@PIPELINES.register_module()
class Depth2Edge(object):
    def __init__(self, strict=True, datasets=['VKITTI_RGBD', 'Hypersim_RGBD']):
        self.strict = strict
        self.datasets = datasets

    def d2e(self, depth):
        H, W = depth.shape
        d = 1.0 / np.clip(depth, 1e-6, None)

        # 简化一个 pad: 传入 (left, right, top, bottom)
        def pad_lr_tb(a, left, right, top, bottom, value=1e6):
            return np.pad(a, ((top, bottom), (left, right)), mode="constant", constant_values=value)
            # return np.pad(a, (top, bottom, left, right), mode="constant", constant_values=value)

        # 2) 按原逻辑构造各种平移版本（对应 F.pad 再切片）
        depth_w2 = pad_lr_tb(d, 1, 0, 0, 0)  # 左1
        depth_w1 = pad_lr_tb(d, 0, 1, 0, 0)  # 右1
        depth_h2 = pad_lr_tb(d, 0, 0, 1, 0)  # 上1
        depth_h1 = pad_lr_tb(d, 0, 0, 0, 1)  # 下1
        depth_rd1 = pad_lr_tb(d, 0, 1, 0, 1)  # 右1, 下1
        depth_ru2 = pad_lr_tb(d, 1, 0, 1, 0)  # 左1, 上1
        depth_ru1 = pad_lr_tb(d, 0, 1, 1, 0)  # 右1, 上1
        depth_lu2 = pad_lr_tb(d, 1, 0, 0, 1)  # 左1, 下1

        # 3) 计算四个方向的“差分”
        dh = depth_h1[1:, :] - depth_h2[:-1, :]
        dw = depth_w1[:, 1:] - depth_w2[:, :-1]
        dru = depth_ru1[:-1, 1:] - depth_lu2[1:, :-1]
        drd = depth_rd1[1:, 1:] - depth_ru2[:-1, :-1]

        # 4) mask（与原代码一致的阈值/逻辑）
        mask1 = (depth_h1[1:, :] < 1e6) & (depth_h2[:-1, :] < 1e6)
        mask2 = (depth_w1[:, 1:] < 1e6) & (depth_w2[:, :-1] < 1e6)
        mask3 = (depth_ru1[:-1, 1:] < 1e6) & (depth_lu2[1:, :-1] < 1e6)
        mask4 = (depth_rd1[1:, 1:] < 1e6) & (depth_ru2[:-1, :-1] < 1e6)

        edge = (np.abs(dh * mask1) +
                np.abs(dw * mask2) +
                np.abs(drd * mask3) +
                np.abs(dru * mask4))  # (H, W)

        valid_cnt = (mask1.astype(np.float32) +
                     mask2.astype(np.float32) +
                     mask3.astype(np.float32) +
                     mask4.astype(np.float32))

        # edge = np.where(valid_cnt > 0, edge / valid_cnt, 0.0).astype(np.float32)
        edge = np.divide(
            edge, valid_cnt,
            out=np.zeros_like(edge, dtype=np.float32),  # 无效处直接写 0
            where=valid_cnt > 0  # 只在有效处做除法
        ).astype(np.float32)

        # 5) 等价于 F.unfold(..., k=3, padding=1) 后做 softmax，并取中心元素（索引4）
        # 先对 edge 做 1 像素的零填充，然后取每个位置的 3x3 邻域（手动 im2col）
        ep = np.pad(edge, 1, mode="constant", constant_values=0.0)
        # 堆叠 3x3 共 9 个邻域平移切片（顺序与 PyTorch unfold 的常见展平一致）
        neigh = np.stack([
            ep[:-2, :-2], ep[:-2, 1:-1], ep[:-2, 2:],
            ep[1:-1, :-2], ep[1:-1, 1:-1], ep[1:-1, 2:],
            ep[2:, :-2], ep[2:, 1:-1], ep[2:, 2:]
        ], axis=-1)  # (H, W, 9)

        # softmax over 最后一个维度
        m = neigh.max(axis=-1, keepdims=True)
        ex = np.exp(neigh - m)
        sm = ex / (ex.sum(axis=-1, keepdims=True) + 1e-12)

        center = sm[..., 4]
        minv = sm.min(axis=-1)
        score = center - minv

        th = np.quantile(score, 0.95)
        out = (score >= th).astype(np.float32)
        return out

    def __call__(self, results):
        if 'dataset' in results.get('img_info', {}):
            if not self.strict or results['img_info']['dataset'] in self.datasets:
                depth = results['depth_gt']
                out = self.d2e(depth)

                results['edge_map'] = out
                results['depth_fields'].append('edge_map')

            else:
                depth = results['depth_gt']
                results['edge_map'] = np.ones_like(depth) * -1
                results['depth_fields'].append('edge_map')
        else:
            depth = results['depth_gt']
            results['edge_map'] = np.ones_like(depth) * -1
            results['depth_fields'].append('edge_map')

        if results.get('mix_depth') is not None:
            if 'dataset' in results.get('mix_img_info', {}):
                if not self.strict or results['mix_img_info']['dataset'] in self.datasets:
                    depth = results['mix_depth']
                    out = self.d2e(depth)

                    results['mix_edge'] = out
                    results['depth_fields'].append('mix_edge')
                else:
                    depth = results['mix_depth']
                    results['mix_edge'] = np.ones_like(depth) * -1
                    results['depth_fields'].append('mix_edge')
            else:
                depth = results['mix_depth']
                results['mix_edge'] = np.ones_like(depth) * -1
                results['depth_fields'].append('mix_edge')

        return results



# @PIPELINES.register_module()
# class ResizeKeepRatio(object):
#     """
#     Resize and pad to a given size. Hold the aspect ratio.
#     This resizing assumes that the camera model remains unchanged.
#     Args:
#         resize_size: predefined output size.
#     """
#
#     def __init__(self, resize_size, padding=None, ignore_label=-1, **kwargs):
#         if isinstance(resize_size, int):
#             self.resize_h = resize_size
#             self.resize_w = resize_size
#         elif isinstance(resize_size, collections.Iterable) and len(resize_size) == 2 \
#                 and isinstance(resize_size[0], int) and isinstance(resize_size[1], int) \
#                 and resize_size[0] > 0 and resize_size[1] > 0:
#             self.resize_h = resize_size[0]
#             self.resize_w = resize_size[1]
#         else:
#             raise (RuntimeError("crop size error.\n"))
#         if padding is None:
#             self.padding = padding
#         elif isinstance(padding, list):
#             if all(isinstance(i, numbers.Number) for i in padding):
#                 self.padding = padding
#             else:
#                 raise (RuntimeError("padding in Crop() should be a number list\n"))
#             if len(padding) != 3:
#                 raise (RuntimeError("padding channel is not equal with 3\n"))
#         else:
#             raise (RuntimeError("padding in Crop() should be a number list\n"))
#         if isinstance(ignore_label, int):
#             self.ignore_label = ignore_label
#         else:
#             raise (RuntimeError("ignore_label should be an integer number\n"))
#         # self.crop_size = kwargs['crop_size']
#         self.canonical_focal = kwargs['focal_length']
#
#     def main_data_transform(self, image, label, intrinsic, cam_model, resize_ratio, padding, to_scale_ratio):
#         """
#         Resize data first and then do the padding.
#         'label' will be scaled.
#         """
#         h, w, _ = image.shape
#         reshape_h = int(resize_ratio * h)
#         reshape_w = int(resize_ratio * w)
#
#         pad_h, pad_w, pad_h_half, pad_w_half = padding
#
#         # resize
#         image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
#         # padding
#         image = cv2.copyMakeBorder(
#             image,
#             pad_h_half,
#             pad_h - pad_h_half,
#             pad_w_half,
#             pad_w - pad_w_half,
#             cv2.BORDER_CONSTANT,
#             value=self.padding)
#
#         if label is not None:
#             # label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
#             label = resize_depth_preserve(label, (reshape_h, reshape_w))
#             label = cv2.copyMakeBorder(
#                 label,
#                 pad_h_half,
#                 pad_h - pad_h_half,
#                 pad_w_half,
#                 pad_w - pad_w_half,
#                 cv2.BORDER_CONSTANT,
#                 value=self.ignore_label)
#             # scale the label
#             label = label / to_scale_ratio
#
#         # Resize, adjust principle point
#         if intrinsic is not None:
#             intrinsic[0] = intrinsic[0] * resize_ratio / to_scale_ratio
#             intrinsic[1] = intrinsic[1] * resize_ratio / to_scale_ratio
#             intrinsic[2] = intrinsic[2] * resize_ratio
#             intrinsic[3] = intrinsic[3] * resize_ratio
#
#         if cam_model is not None:
#             # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
#             cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
#             cam_model = cv2.copyMakeBorder(
#                 cam_model,
#                 pad_h_half,
#                 pad_h - pad_h_half,
#                 pad_w_half,
#                 pad_w - pad_w_half,
#                 cv2.BORDER_CONSTANT,
#                 value=self.ignore_label)
#
#         # Pad, adjust the principle point
#         if intrinsic is not None:
#             intrinsic[2] = intrinsic[2] + pad_w_half
#             intrinsic[3] = intrinsic[3] + pad_h_half
#         return image, label, intrinsic, cam_model
#
#     def __call__(self, results):
#         img = results['img']
#
#         img_shape = img.shape
#         results['img'] = img
#         results['img_shape'] = img_shape
#
#         # crop depth
#         for key in results.get('depth_fields', []):
#             results[key] = self.crop(results[key], crop_bbox)
#
#         results["depth_shape"] = img_shape
#
#         target_h, target_w, _ = img.shape
#         resize_ratio_h = self.resize_h / target_h
#         resize_ratio_w = self.resize_w / target_w
#         resize_ratio = min(resize_ratio_h, resize_ratio_w)
#         reshape_h = int(resize_ratio * target_h)
#         reshape_w = int(resize_ratio * target_w)
#         pad_h = max(self.resize_h - reshape_h, 0)
#         pad_w = max(self.resize_w - reshape_w, 0)
#         pad_h_half = int(pad_h / 2)
#         pad_w_half = int(pad_w / 2)
#
#         pad_info = [pad_h, pad_w, pad_h_half, pad_w_half]
#         to_scale_ratio = self.get_label_scale_factor(images[0], intrinsics[0], resize_ratio)
#
#
#
#         for i in range(len(images)):
#             img = images[i]
#             label = labels[i] if i < len(labels) else None
#             intrinsic = intrinsics[i] if i < len(intrinsics) else None
#             cam_model = cam_models[i] if cam_models is not None and i < len(cam_models) else None
#             img, label, intrinsic, cam_model = self.main_data_transform(
#                 img, label, intrinsic, cam_model, resize_ratio, pad_info, to_scale_ratio)
#             images[i] = img
#             if label is not None:
#                 labels[i] = label
#             if intrinsic is not None:
#                 intrinsics[i] = intrinsic
#             if cam_model is not None:
#                 cam_models[i] = cam_model
#
#         if other_labels is not None:
#
#             for i, other_lab in enumerate(other_labels):
#                 # resize
#                 other_lab = cv2.resize(other_lab, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
#                 # pad
#                 other_labels[i] = cv2.copyMakeBorder(
#                     other_lab,
#                     pad_h_half,
#                     pad_h - pad_h_half,
#                     pad_w_half,
#                     pad_w - pad_w_half,
#                     cv2.BORDER_CONSTANT,
#                     value=self.ignore_label)
#
#         pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
#         if transform_paras is not None:
#             pad_old = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
#             new_pad = [pad_old[0] + pad[0], pad_old[1] + pad[1], pad_old[2] + pad[2], pad_old[3] + pad[3]]
#             transform_paras.update(dict(pad=new_pad))
#             if 'label_scale_factor' in transform_paras:
#                 transform_paras['label_scale_factor'] = transform_paras['label_scale_factor'] * 1.0 / to_scale_ratio
#             else:
#                 transform_paras.update(label_scale_factor=1.0 / to_scale_ratio)
#         return images, labels, intrinsics, cam_models, other_labels, transform_paras

@PIPELINES.register_module()
class MD2EDepth2Edge(object):
    def __init__(self, quantile=0.9, eps=1e-6, apply_to_mix=True):
        self.quantile = quantile
        self.eps = eps
        self.apply_to_mix = apply_to_mix

    def d2e(self, depth):
        depth = np.asarray(depth, dtype=np.float32)
        valid = depth > self.eps
        inv_depth = np.zeros_like(depth, dtype=np.float32)
        inv_depth[valid] = 1.0 / np.clip(depth[valid], self.eps, None)
        H, W = inv_depth.shape

        def pad_lr_tb(a, left, right, top, bottom):
            return np.pad(a, ((top, bottom), (left, right)), mode='constant', constant_values=0.0)

        def pad_mask(a, left, right, top, bottom):
            return np.pad(a, ((top, bottom), (left, right)), mode='constant', constant_values=False)

        depth_w2 = pad_lr_tb(inv_depth, 1, 0, 0, 0)
        depth_w1 = pad_lr_tb(inv_depth, 0, 1, 0, 0)
        depth_h2 = pad_lr_tb(inv_depth, 0, 0, 1, 0)
        depth_h1 = pad_lr_tb(inv_depth, 0, 0, 0, 1)
        depth_rd1 = pad_lr_tb(inv_depth, 0, 1, 0, 1)
        depth_ru2 = pad_lr_tb(inv_depth, 1, 0, 1, 0)
        depth_ru1 = pad_lr_tb(inv_depth, 0, 1, 1, 0)
        depth_lu2 = pad_lr_tb(inv_depth, 1, 0, 0, 1)

        valid_w2 = pad_mask(valid, 1, 0, 0, 0)
        valid_w1 = pad_mask(valid, 0, 1, 0, 0)
        valid_h2 = pad_mask(valid, 0, 0, 1, 0)
        valid_h1 = pad_mask(valid, 0, 0, 0, 1)
        valid_rd1 = pad_mask(valid, 0, 1, 0, 1)
        valid_ru2 = pad_mask(valid, 1, 0, 1, 0)
        valid_ru1 = pad_mask(valid, 0, 1, 1, 0)
        valid_lu2 = pad_mask(valid, 1, 0, 0, 1)

        dh = depth_h1[1:, :] - depth_h2[:-1, :]
        dw = depth_w1[:, 1:] - depth_w2[:, :-1]
        dru = depth_ru1[:-1, 1:] - depth_lu2[1:, :-1]
        drd = depth_rd1[1:, 1:] - depth_ru2[:-1, :-1]

        mask1 = valid_h1[1:, :] & valid_h2[:-1, :]
        mask2 = valid_w1[:, 1:] & valid_w2[:, :-1]
        mask3 = valid_ru1[:-1, 1:] & valid_lu2[1:, :-1]
        mask4 = valid_rd1[1:, 1:] & valid_ru2[:-1, :-1]

        saliency = (np.abs(dh) * mask1.astype(np.float32) +
                    np.abs(dw) * mask2.astype(np.float32) +
                    np.abs(dru) * mask3.astype(np.float32) +
                    np.abs(drd) * mask4.astype(np.float32))

        ep = np.pad(saliency, 1, mode='constant', constant_values=0.0)
        neigh = np.stack([
            ep[:-2, :-2], ep[:-2, 1:-1], ep[:-2, 2:],
            ep[1:-1, :-2], ep[1:-1, 1:-1], ep[1:-1, 2:],
            ep[2:, :-2], ep[2:, 1:-1], ep[2:, 2:]
        ], axis=-1)
        neigh = neigh - neigh.max(axis=-1, keepdims=True)
        exp_neigh = np.exp(neigh)
        prob = exp_neigh / (exp_neigh.sum(axis=-1, keepdims=True) + 1e-12)
        score = prob[..., 4] - prob.min(axis=-1)

        valid_score = score[valid]
        if valid_score.size == 0:
            return np.zeros((H, W), dtype=np.float32)
        threshold = np.quantile(valid_score, self.quantile)
        return ((score >= threshold) & valid).astype(np.float32)

    def _append_depth_field(self, results, key):
        if 'depth_fields' not in results:
            results['depth_fields'] = []
        if key not in results['depth_fields']:
            results['depth_fields'].append(key)

    def __call__(self, results):
        if 'depth_gt' in results:
            results['edge_map'] = self.d2e(results['depth_gt'])
            self._append_depth_field(results, 'edge_map')

        if self.apply_to_mix and results.get('mix_depth') is not None:
            results['mix_edge'] = self.d2e(results['mix_depth'])
            self._append_depth_field(results, 'mix_edge')

        return results
