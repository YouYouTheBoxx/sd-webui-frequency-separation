import math
import torch

class FrequencySplit:
    """Split an image tensor into low, mid, and high frequency bands."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "low_end": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0}),
                "mid_start": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "mid_end": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                "high_start": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
                "high_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    FUNCTION = "split"
    CATEGORY = "frequency"

    @staticmethod
    def _image_to_freq(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return torch.fft.fftn(image, dim=(-2, -1))

    @staticmethod
    def _freq_to_image(freq: torch.Tensor) -> torch.Tensor:
        img = torch.fft.ifftn(freq, dim=(-2, -1)).real
        return img

    @staticmethod
    def _mask(shape, fr, overlap):
        h, w = shape
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.float() - center_h
        x = x.float() - center_w
        max_freq = math.sqrt(center_h ** 2 + center_w ** 2)
        freq_mag = torch.sqrt(x ** 2 + y ** 2) / max_freq
        low, high = fr
        trans = overlap
        lower = torch.sigmoid((freq_mag - low) / trans * 10)
        upper = torch.sigmoid((high - freq_mag) / trans * 10)
        return lower * upper

    def split(self, image, low_start, low_end, mid_start, mid_end, high_start, high_end, overlap):
        freq = self._image_to_freq(image)
        shape = freq.shape[-2:]
        low_mask = self._mask(shape, (low_start, low_end), overlap).to(freq.device)
        mid_mask = self._mask(shape, (mid_start, mid_end), overlap).to(freq.device)
        high_mask = self._mask(shape, (high_start, high_end), overlap).to(freq.device)
        low = self._freq_to_image(freq * low_mask.unsqueeze(0).unsqueeze(0))
        mid = self._freq_to_image(freq * mid_mask.unsqueeze(0).unsqueeze(0))
        high = self._freq_to_image(freq * high_mask.unsqueeze(0).unsqueeze(0))
        return (low, mid, high)

class FrequencyRecombine:
    """Recombine frequency bands into a single image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low": ("IMAGE",),
                "mid": ("IMAGE",),
                "high": ("IMAGE",),
                "low_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "low_end": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0}),
                "mid_start": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "mid_end": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                "high_start": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
                "high_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recombine"
    CATEGORY = "frequency"

    @staticmethod
    def _image_to_freq(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return torch.fft.fftn(image, dim=(-2, -1))

    @staticmethod
    def _freq_to_image(freq: torch.Tensor) -> torch.Tensor:
        img = torch.fft.ifftn(freq, dim=(-2, -1)).real
        return img

    @staticmethod
    def _mask(shape, fr, overlap):
        h, w = shape
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.float() - center_h
        x = x.float() - center_w
        max_freq = math.sqrt(center_h ** 2 + center_w ** 2)
        freq_mag = torch.sqrt(x ** 2 + y ** 2) / max_freq
        low, high = fr
        trans = overlap
        lower = torch.sigmoid((freq_mag - low) / trans * 10)
        upper = torch.sigmoid((high - freq_mag) / trans * 10)
        return lower * upper

    def recombine(self, low, mid, high, low_start, low_end, mid_start, mid_end, high_start, high_end, overlap):
        low_f = self._image_to_freq(low)
        mid_f = self._image_to_freq(mid)
        high_f = self._image_to_freq(high)
        shape = low_f.shape[-2:]
        low_mask = self._mask(shape, (low_start, low_end), overlap).to(low_f.device)
        mid_mask = self._mask(shape, (mid_start, mid_end), overlap).to(low_f.device)
        high_mask = self._mask(shape, (high_start, high_end), overlap).to(low_f.device)
        combined = low_f * low_mask.unsqueeze(0).unsqueeze(0) + \
                   mid_f * mid_mask.unsqueeze(0).unsqueeze(0) + \
                   high_f * high_mask.unsqueeze(0).unsqueeze(0)
        image = self._freq_to_image(combined)
        return (image,)
