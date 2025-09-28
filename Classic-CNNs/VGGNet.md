# VGGNet Architecture & Implementation

VGGNet (2014) by the Visual Geometry Group (Oxford) proved that depth is a critical component for good performance. They standardized the architecture by exclusively using very small **3x3 convolutional filters** stacked on top of each other.

## 🧠 Architecture Overview (VGG-16)
VGG-16 consists of 16 weight layers (13 convolutional layers and 3 fully connected layers):
- **Input:** 224x224 RGB image
- **Conv Blocks:** Uses stacks of 3x3 convolutions with stride 1 and padding 1 (to preserve spatial dimension), followed by a 2x2 Max Pooling layer (stride 2) to halve the spatial resolution.
- As the spatial resolution halves, the number of channels (filters) doubles (64 -> 128 -> 256 -> 512 -> 512).
- **Classifier:** Ends with three Fully Connected layers (4096 -> 4096 -> 1000).

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        # Helper function to create conv blocks
        def conv_block(in_c, out_c, num_convs):
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_c = out_c
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            return nn.Sequential(*layers)
            
        self.features = nn.Sequential(
            conv_block(3, 64, 2),    # Block 1
            conv_block(64, 128, 2),  # Block 2
            conv_block(128, 256, 3), # Block 3
            conv_block(256, 512, 3), # Block 4
            conv_block(512, 512, 3), # Block 5
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Basic Run / Testing the model
if __name__ == "__main__":
    model = VGG16(num_classes=1000)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 1000]
```
