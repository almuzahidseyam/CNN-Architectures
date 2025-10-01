# YOLO (You Only Look Once) Architecture

YOLO (2016) by Joseph Redmon et al. reframed object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities, achieving real-time speeds.

## 🧠 Architecture Overview (YOLOv1)
Unlike R-CNN which uses regional proposals (a two-stage pipeline), YOLO passes the image through the network exactly once.
1. **Grid System:** The input image is divided into an $S \times S$ grid (e.g., 7x7).
2. **Responsibility:** If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
3. **Predictions:** Each grid cell predicts $B$ bounding boxes and their confidence scores, as well as $C$ conditional class probabilities.
4. **Network:** The original YOLO model uses a modified GoogLeNet architecture with 24 convolutional layers followed by 2 fully connected layers.

---

## 💻 PyTorch Implementation (YOLO Loss/Head Concept)

```python
import torch
import torch.nn as nn

class YoloV1Head(nn.Module):
    """
    Conceptual implementation of the YOLOv1 fully connected head.
    Assumes feature extractor has output shape [Batch, 1024, 7, 7].
    """
    def __init__(self, S=7, B=2, C=20):
        super(YoloV1Head, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            # Output: S * S * (B * 5 + C)
            # 5 values per box (x, y, w, h, confidence)
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        # x is the output from the conv backbone (e.g., [batch, 1024, 7, 7])
        predictions = self.fc(x)
        # Reshape to [batch, S, S, B*5 + C]
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        return predictions

# Basic Run / Testing the model
if __name__ == "__main__":
    # S=7 grid, B=2 boxes, C=20 classes (Pascal VOC)
    model = YoloV1Head(S=7, B=2, C=20)
    dummy_features = torch.randn(1, 1024, 7, 7)
    output = model(dummy_features)
    # Output shape: [1, 7, 7, 30] where 30 = (2*5) + 20
    print(f"Output shape: {output.shape}") 
```
