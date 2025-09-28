# ResNet (Residual Network) Architecture & Implementation

Introduced by Kaiming He et al. in 2015, **ResNet** revolutionized deep learning by allowing the training of extremely deep networks (100+ layers) without suffering from the vanishing gradient problem.

## 🧠 Architecture Overview (ResNet-18)
The core innovation of ResNet is the **Residual Block** (Skip Connection), which adds the original input $x$ to the output of the convolutional layers $F(x)$. So the layer learns the *residual* $F(x) = H(x) - x$ instead of the underlying mapping $H(x)$.

**ResNet-18 Structure:**
1. **Conv1:** 7x7 Conv (64 filters, stride 2) + MaxPooling (3x3, stride 2)
2. **Layer 1:** 2x Basic Blocks (64 channels)
3. **Layer 2:** 2x Basic Blocks (128 channels, downsampling)
4. **Layer 3:** 2x Basic Blocks (256 channels, downsampling)
5. **Layer 4:** 2x Basic Blocks (512 channels, downsampling)
6. **Classifier:** Global Average Pooling + Fully Connected Layer (1000 classes)

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Main Path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Adding the skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Basic Run / Testing the model
if __name__ == "__main__":
    model = ResNet18(num_classes=10) # Testing for 10 classes
    # Dummy input: Batch size of 1, 3 color channels, 224x224 dimensions
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 10]
```
