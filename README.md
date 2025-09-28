# 🧠 CNN Architectures & Variants

A comprehensive collection and timeline of popular **Convolutional Neural Network (CNN)** architectures. This repository serves as a learning hub to track the evolution of CNNs from classic models to modern hybrid approaches.

---

## 📌 Classic CNN Architectures
*The foundational models that pioneered deep learning in computer vision.*

- **LeNet-5 (1998)** – The first successful CNN (handwritten digit recognition).
- **AlexNet (2012)** – Breakthrough in ImageNet; introduced deeper networks, ReLU, dropout, and data augmentation.
- **ZFNet (2013)** – Improved AlexNet by visualizing filters and tweaking hyperparameters.
- **VGGNet (2014)** – Very deep architecture (16–19 layers) using uniform 3×3 kernels.
- **GoogLeNet / Inception v1 (2014)** – Introduced Inception modules (multi-scale convolutions).

---

## 📌 Deeper & Efficient CNN Variants
*Optimized for better performance, deeper layers, and mobile efficiency.*

- **Inception v2/v3 (2015)** – Factorized convolutions, batch normalization.
- **Inception v4 (2016)** – Even deeper, combined with residual connections.
- **ResNet (2015)** – Introduced Skip connections (residual learning), enabling the training of 100+ layers.
- **Wide ResNet (2016)** – Shallower but wider ResNet variant.
- **ResNeXt (2017)** – Grouped convolutions, introducing cardinality as a new dimension.
- **DenseNet (2017)** – Dense connections ensuring feature reuse across layers.
- **Xception (2017)** – Depthwise separable convolutions (an extreme version of Inception).
- **MobileNet v1/v2/v3 (2017–2019)** – Lightweight CNNs explicitly designed for mobile/embedded devices.
- **ShuffleNet (2017)** – Channel shuffling for highly efficient inference.
- **SqueezeNet (2016)** – Extremely small model utilizing "fire modules".
- **EfficientNet (2019)** – Compound scaling methodology (balancing depth, width, and resolution).

---

## 📌 Detection & Segmentation-Oriented Variants
*Specialized architectures for finding objects and segmenting pixels.*

- **R-CNN (2014)** – Region-based CNN for object detection.
- **Fast R-CNN (2015)** – Faster, optimized version of R-CNN.
- **Faster R-CNN (2015)** – Introduced Region Proposal Networks (RPN) for real-time proposals.
- **Mask R-CNN (2017)** – Extended Faster R-CNN by adding pixel-level segmentation masks.
- **YOLO (2016–present)** – Real-time, single-pass object detection CNN (multiple versions available).
- **SSD (Single Shot Detector, 2016)** – Fast one-shot object detection CNN.
- **U-Net (2015)** – Encoder-decoder CNN primarily for biomedical image segmentation.
- **SegNet (2015)** – Semantic segmentation using a deep encoder-decoder CNN.
- **FCN (Fully Convolutional Network, 2015)** – Pixel-wise prediction CNN.

---

## 📌 Modern/Hybrid CNN Approaches
*Next-generation architectures blending CNNs with Transformer concepts.*

- **RegNet (2020)** – Network design driven by design space exploration.
- **ConvNeXt (2022)** – CNN completely redesigned with Transformer-style improvements.
- **RepVGG (2021)** – VGG-style inference with structural re-parameterization during training.
- **EfficientNetV2 (2021)** – Faster training and better parameter efficiency compared to the original EfficientNet.
- **HRNet (2019)** – Maintains high-resolution representations throughout the network for accurate segmentation.

---

## 📂 Repository Structure
- `Classic-CNNs/`
- `Efficient-CNNs/`
- `Detection-Segmentation/`
- `Modern-Hybrid-CNNs/`

*More implementations and code snippets coming soon!*
