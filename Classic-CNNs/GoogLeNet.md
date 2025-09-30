# GoogLeNet (Inception v1) Architecture & Implementation

GoogLeNet (2014) won the ILSVRC-2014 competition. It introduced the **Inception Module**, which performs multi-scale convolutions (1x1, 3x3, 5x5) and max pooling in parallel within the same layer, allowing the network to learn both local and global features efficiently.

## 🧠 Architecture Overview
- **Inception Module:** The building block. It concatenates the outputs of 1x1, 3x3, 5x5 convolutions and a 3x3 max pooling operation.
- **1x1 Convolutions:** Used extensively as bottleneck layers to reduce dimensionality (number of channels) before the expensive 3x3 and 5x5 convolutions, saving massive computational cost.
- **Global Average Pooling:** Used at the end of the network instead of fully connected layers, drastically reducing the number of parameters.
- **Auxiliary Classifiers:** Connected to intermediate layers to encourage gradient flow during training (combating the vanishing gradient problem in deep networks).

---

## 💻 PyTorch Implementation (Inception Module)

```python
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 branch
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 branch
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2) # 5x5 is often approximated by two 3x3 in later versions
        )
        
        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate outputs along the channel dimension
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# Example Usage
if __name__ == "__main__":
    # Inception 3a from GoogLeNet
    model = InceptionModule(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
    dummy_input = torch.randn(1, 192, 28, 28)
    output = model(dummy_input)
    # Output channels = 64 + 128 + 32 + 32 = 256
    print(f"Output shape: {output.shape}") # Expected: [1, 256, 28, 28]
```
