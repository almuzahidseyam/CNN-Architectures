# MobileNetV1 Architecture & Implementation

MobileNet (2017) by Google introduced **Depthwise Separable Convolutions** to drastically reduce the number of parameters and computational cost, making deep learning models feasible for mobile and embedded vision applications.

## 🧠 Architecture Overview
Instead of standard convolutions (which apply a single filter across all input channels and spatial dimensions simultaneously), MobileNet splits the operation into two layers:
1. **Depthwise Convolution:** Applies a single filter to each input channel independently.
2. **Pointwise Convolution:** A 1x1 convolution that combines the outputs of the depthwise convolution.

This reduces computation by a factor of roughly `1/N + 1/D^2`, where N is the number of output channels and D is the spatial dimension of the kernel (usually 3).

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 1. Depthwise Convolution (groups=in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Pointwise Convolution (1x1 conv)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        
        # Standard initial conv layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stack of Depthwise Separable Convolutions
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            # 5 blocks of 512
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Basic Run / Testing the model
if __name__ == "__main__":
    model = MobileNetV1(num_classes=1000)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 1000]
```
