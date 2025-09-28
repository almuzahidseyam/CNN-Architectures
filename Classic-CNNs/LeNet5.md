# LeNet-5 Architecture & Implementation

LeNet-5 (1998) by Yann LeCun is one of the earliest convolutional neural networks and promoted the development of deep learning. It was primarily designed for handwritten and machine-printed character recognition (MNIST).

## 🧠 Architecture Overview
The network has 7 layers (excluding the input layer):
1. **Input:** 32x32 grayscale image (1 channel)
2. **C1 (Convolution):** 6 filters (5x5), stride 1. Output: 28x28x6
3. **S2 (Average Pooling/Subsampling):** 2x2, stride 2. Output: 14x14x6
4. **C3 (Convolution):** 16 filters (5x5), stride 1. Output: 10x10x16
5. **S4 (Average Pooling):** 2x2, stride 2. Output: 5x5x16
6. **C5 (Fully Connected/Conv):** 120 neurons
7. **F6 (Fully Connected):** 84 neurons
8. **Output:** 10 neurons (Softmax for digits 0-9)

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten for dense layers
        x = self.classifier(x)
        return x

# Basic Run / Testing the model
if __name__ == "__main__":
    model = LeNet5()
    # Dummy input: Batch size of 1, 1 color channel, 32x32 dimensions
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 10]
```
