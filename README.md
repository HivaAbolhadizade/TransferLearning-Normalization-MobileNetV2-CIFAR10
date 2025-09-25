# Transfer Learning Optimization on MobileNetV2 for CIFAR-10

📌 **Overview**
This project investigates the impact of three normalization techniques — **Batch Normalization (BatchNorm)**, **Layer Normalization (LayerNorm)**, and **Filter Response Normalization (FRN)** — in the context of **Transfer Learning** using the **MobileNetV2** architecture on the **CIFAR-10** dataset.
Additionally, the effect of **Gradient Clipping** on gradient stability and convergence is analyzed.
Experiments reveal that **FRN without Clipping** achieves the highest accuracy, while **Gradient Clipping** improves convergence for **BatchNorm**. LayerNorm also shows stable performance without clipping.

---
## Project Structure

- **Instruction**: Contains the project problem statement.  
- **Solution**: Contains all implementation files, experiments, and the report.

---
🔹 **Normalization Techniques**

* **BatchNorm**: Normalizes features across the batch dimension. Sensitive to small batch sizes.
* **LayerNorm**: Normalizes across feature dimensions within each sample. Independent of batch statistics.
* **FRN**: Normalizes independently of batch or channel statistics, making it robust to small batch sizes.

---

🔹 **Gradient Clipping**

* Threshold: `1.0`
* Purpose: Prevent gradient explosion and improve stability in fine-tuning.
* Observations: Positive effect for BatchNorm, minimal or neutral effect for LayerNorm and FRN.

---

🔹 **Dataset**

* **CIFAR-10**

  * 60,000 color images, 32×32 pixels, 10 classes.
  * Resized to 224×224 for MobileNetV2 input.
  * Data augmentation for training:

    * `RandomHorizontalFlip`
    * `RandomCrop` with padding=4
    * Normalization with CIFAR-10 mean/std
  * Train/Validation split: 80% / 20%

---

🔹 **Model Architecture**

* **Base Model**: MobileNetV2 pretrained on ImageNet (all layers frozen except final block `features.18`).
* **Custom Heads**:

  1. **BatchNormHead**: FC(256) → BatchNorm → ReLU → Dropout → FC(10)
  2. **LayerNormHead**: Same as above, replacing BatchNorm with LayerNorm.
  3. **FRNHead**: FC(256) → FRN → ReLU → Dropout → FC(10)

---

🔹 **Training Configuration**

* Loss: **CrossEntropyLoss**
* Optimizer: **Adam**
* Learning rates: `1e-3` (head), `1e-5` (last MobileNetV2 block)
* Epochs: **15**
* Batch size: **64** (32 for gradient analysis)
* 6 experiments: 3 normalization heads × (Clipping / No Clipping)

---

🔹 **Results**

| Normalization | Clipping | Final Accuracy | Effect of Clipping  |
| ------------- | -------- | -------------- | ------------------- |
| BatchNorm     | No       | 89.64%         | –                   |
| BatchNorm     | Yes      | **90.71%**     | ✅ Positive (+1.07%) |
| LayerNorm     | No       | 90.22%         | –                   |
| LayerNorm     | Yes      | 90.15%         | ⚠ Neutral           |
| FRN           | No       | **90.57%**     | –                   |
| FRN           | Yes      | 90.41%         | ⚠ Neutral           |

**Key Findings**:

* FRN without clipping delivers the best accuracy.
* Gradient Clipping benefits BatchNorm the most.
* LayerNorm and FRN remain stable without clipping.

---

🔹 **Implementation Highlights**

* Custom FRN layer implemented in PyTorch.
* Gradient norm tracking across epochs.
* Loss landscape and gradient distribution analysis.
* Data loader class (`CIFAR10DataLoader`) with built-in visualization.

---

✅ **Conclusion**

* **FRN-NoClip** is the most stable and accurate configuration for this setup.
* For small batch sizes, FRN and LayerNorm are strong candidates without the need for clipping.
* BatchNorm benefits significantly from Gradient Clipping during fine-tuning.
