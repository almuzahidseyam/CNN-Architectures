# AlexNet Architecture & Implementation

AlexNet (2012) by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton brought a breakthrough in computer vision by winning the ImageNet ILSVRC-2012 competition. It popularized the use of **ReLU** activations, **Dropout** for regularization, and Data Augmentation.

## 🧠 Architecture Overview
The network has 8 learned layers (5 convolutional and 3 fully connected):
1. **Input:** 227x227 RGB image (3 channels).
2. **Conv1:** 96 filters (11x11), stride 4 + ReLU + MaxPool(3x3, stride 2).
3. **Conv2:** 256 filters (5x5), padding 2 + ReLU + MaxPool.
4. **Conv3:** 384 filters (3x3), padding 1 + ReLU.
5. **Conv4:** 384 filters (3x3), padding 1 + ReLU.
6. **Conv5:** 256 filters (3x3), padding 1 + ReLU + MaxPool.
7. **FC6:** 4096 neurons + ReLU + Dropout(0.5).
8. **FC7:** 4096 neurons + ReLU + Dropout(0.5).
9. **Output (FC8):** 1000 neurons (Softmax for ImageNet classes).

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Basic Run / Testing the model
if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    dummy_input = torch.randn(1, 3, 227, 227)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 1000]
```
