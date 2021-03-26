# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import PIL.Image

from .file_utils import _is_torch, is_torch_available


def is_torch_tensor(obj):
    return _is_torch(obj) if is_torch_available() else False


# In the future we can add a TF implementation here when we have TF models.
class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """

    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`torch.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        Converts :obj:`image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last
        axis if needed.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`numpy.ndarray` or :obj:`torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (:obj:`bool`, `optional`):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to :obj:`True` if the image type is a floating type, :obj:`False` otherwise.
        """
        self._ensure_format_supported(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts :obj:`image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (:obj:`bool`, `optional`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to :obj:`True` if the image is a PIL Image or an array/tensor of integers, :obj:`False`
                otherwise.
            channel_first (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if rescale is None:
            rescale = isinstance(image.flat[0], np.integer)

        if rescale:
            image = image.astype(np.float32) / 255.0

        if channel_first:
            image = image.transpose(2, 0, 1)

        return image

    def normalize(self, image, mean, std):
        """
        Normalizes :obj:`image` with :obj:`mean` and :obj:`std`. Note that this will trigger a conversion of
        :obj:`image` to a NumPy array if it's a PIL Image.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to normalize.
            mean (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean)
            if not isinstance(std, np.ndarray):
                std = np.array(std)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def resize(self, image, size, resample=PIL.Image.BILINEAR):
        """
        Resizes :obj:`image`. Note that this will trigger a conversion of :obj:`image` to a PIL Image.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to use for resizing the image.
            resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
                The filter to user for resampling.
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            size = (size, size)
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.resize(size, resample=resample)
