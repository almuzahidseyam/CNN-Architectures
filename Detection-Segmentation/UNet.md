# U-Net Architecture & Implementation

U-Net (2015) was designed specifically for **biomedical image segmentation**. Its unique "U-shaped" architecture relies heavily on data augmentation to use available annotated samples efficiently.

## 🧠 Architecture Overview
The architecture consists of a contracting path (encoder) and an expansive path (decoder).
1. **Encoder (Contracting Path):** Acts like a traditional CNN, capturing context. It repeatedly applies two 3x3 convolutions (followed by ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
2. **Decoder (Expansive Path):** Symmetrically expands the spatial dimensions to allow precise localization. It uses up-convolutions (transpose convolutions), concatenates with the correspondingly cropped feature map from the contracting path (Skip Connections), and applies two 3x3 convolutions.
3. **Skip Connections:** The key feature of U-Net. High-resolution features from the encoder are concatenated with the upsampled features in the decoder, providing the necessary spatial information that was lost during pooling.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=2):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512) # 512 from upconv + 512 from skip
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)
        
        # Final output convolution
        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        
        # Bottleneck
        x_bottleneck = self.bottleneck(self.pool4(x4))
        
        # Decoder passes with skip connections (concatenation over channel dim)
        x = self.upconv4(x_bottleneck)
        x = self.up4(torch.cat([x, x4], dim=1))
        
        x = self.upconv3(x)
        x = self.up3(torch.cat([x, x3], dim=1))
        
        x = self.upconv2(x)
        x = self.up2(torch.cat([x, x2], dim=1))
        
        x = self.upconv1(x)
        x = self.up1(torch.cat([x, x1], dim=1))
        
        logits = self.out_conv(x)
        return logits

# Basic Run / Testing the model
if __name__ == "__main__":
    model = UNet(in_channels=1, out_classes=2) # e.g., grayscale medical image to binary mask
    dummy_input = torch.randn(1, 1, 256, 256) # Batch=1, Channel=1, HxW=256x256
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [1, 2, 256, 256]
```
