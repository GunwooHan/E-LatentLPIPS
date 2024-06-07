import torch.nn as nn
import torch

import torchvision


class ContrastAdjust(nn.Module):
    def __init__(self, contrast_range=0):
        super().__init__()
        # assert len(contrast_range) == 2, "Input tensor must include min, max value"
        if len(contrast_range) == 1:
            self.contrast_range = [0, contrast_range]
        elif len(contrast_range) == 2:
            self.contrast_range = contrast_range
        else:
            raise ValueError("Input tensor must include min, max value, only support 1 or 2 dimenstion")
        self.contrast_range = contrast_range

    def __call__(self, input_tensor):
        batch_size, channels, height, width = input_tensor.shape
        self.contrast_factor = torch.rand(1) * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[
            0]

        output_tensor = torch.empty_like(input_tensor)
        for i in range(batch_size):
            img = input_tensor[i]
            mean = img.mean([1, 2], keepdim=True)
            adjusted_img = (img - mean) * self.contrast_factor + mean
            output_tensor[i] = adjusted_img.clamp(min=-1, max=1)

        return output_tensor


class BrightnessAdjust(nn.Module):
    def __init__(self, brightness_range=(0, 0.5)):
        super().__init__()
        # assert len(contrast_range) == 2, "Input tensor must include min, max value"
        if len(brightness_range) == 1:
            self.brightness_range = [0, brightness_range]
        elif len(brightness_range) == 2:
            self.brightness_range = brightness_range
        else:
            raise ValueError("Input tensor must include min, max value, only support 1 or 2 dimenstion")
        self.brightness_range = brightness_range

    def __call__(self, input_tensor):
        batch_size, channels, height, width = input_tensor.shape
        self.brightness_factor = torch.rand(1) * (self.brightness_range[1] - self.brightness_range[0]) + \
                                 self.brightness_range[0]

        output_tensor = torch.empty_like(input_tensor)
        for i in range(batch_size):
            img = input_tensor[i]
            max_value = img.max()
            adjusted_img = img * self.brightness_factor
            output_tensor[i] = adjusted_img.clamp(max=max_value)

        return output_tensor


if __name__ == '__main__':
    inputs = torch.randn(1, 4, 64, 64)

    outputs = []

    # transform = ContrastAdjust([0, 1])
    transform = BrightnessAdjust([0, 1])
    for i in range(8):
        outputs.append(transform(inputs))

    torchvision.utils.save_image(torch.cat([inputs] + outputs, dim=0), "aug_comp.png", nrow=9, normalize=True)
