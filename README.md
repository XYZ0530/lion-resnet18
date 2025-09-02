# lion-resnet18
ResNet-18 on CIFAR-10 with LION optimizer; +2.7% acc vs Adam and ~33% faster training.

# Lion-ResNet18: Optimizing Image Classification with the Lion Optimizer

## 📌 Overview

This project investigates the effectiveness of the **Lion optimizer** on the **ResNet-18 architecture** for image classification using the **CIFAR-10 dataset**.
While Adam is a widely used baseline optimizer, Lion introduces a **sign-based update rule** that is both **memory-efficient** and **faster in convergence**.

Our experiments demonstrate that Lion not only improves accuracy but also reduces training time significantly compared to Adam.

---

## 🚀 Key Results

* **Accuracy**:

  * ResNet-18 + Lion → **81.16%**
  * ResNet-18 + Adam → **78.40%**
* **Training Time**:

  * Lion converged **32.91% faster** (≈ 270.62s vs 403.37s on average).
* **Stability**: Lion produced more consistent results across multiple trials.
* **Efficiency**: Achieved higher accuracy with \~2× fewer epochs.

---

## 🔬 Methodology

1. **Architecture**:

   * Base model: **ResNet-18** (18-layer residual network with skip connections).
   * No structural modifications, only optimizer swap to maintain fairness.

2. **Dataset**:

   * **CIFAR-10** → 60k images (32×32, 10 classes).
   * Split: 50k train, 10k test.

3. **Optimizers**:

   * **Adam**: Baseline, LR = 1e-3.
   * **Lion**: Custom PyTorch implementation, LR = 1e-4 (lower to stabilize sign-based updates).

4. **Training Setup**:

   * Early stopping based on validation loss.
   * Cross-validation and repeated runs (×5) for statistical robustness.
   * Metrics: Accuracy, precision, recall, confusion matrix.

5. **Implementation Details**:

   * `lion_optimizer.py` → Custom PyTorch optimizer implementation.
   * Training notebooks (`Train2.ipynb` … `Train5.ipynb`) include experimental runs and results.
   * Final evaluation in `Model_final.ipynb`.

---

## 📈 Visual Insights

* **Confusion Matrices**: Lion reduced misclassification across most CIFAR-10 classes.
* **Loss & Accuracy Curves**: Lion showed faster convergence and smoother stability.
* **Training Time Bar Charts**: Clear evidence of reduced compute time per run.

---

## 🛠️ Repository Structure

```
├── lion_optimizer.py      # Custom PyTorch Lion optimizer
├── Model_final.ipynb      # Final evaluation notebook
├── Train2.ipynb           # Training run with Lion vs Adam
├── Train3.ipynb
├── Train4.ipynb
├── Train5.ipynb
├── Report.pdf             # Full project report with methodology & results
└── README.md              # Project documentation
```

---

## ⚡ Installation & Usage

### Requirements

* Python 3.8+
* PyTorch ≥ 1.12
* torchvision
* matplotlib, seaborn (for visualizations)

### Setup

```bash
git clone https://github.com/<your-username>/lion-resnet18.git
cd lion-resnet18
pip install -r requirements.txt
```

### Training

```python
# Example: Training ResNet-18 with Lion
from lion_optimizer import Lion
import torch.optim as optim

model = ...  # ResNet-18
optimizer = Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-2)
```

Or simply open one of the training notebooks (`TrainX.ipynb`) to reproduce results.

---

## 🎯 Contributions

* **Tan Yi Zhao** (Leader): Implemented ResNet-18 with Lion, training & preprocessing, final integration.
* **Sim Wen Ken**: Data preparation, model architecture explanation.
* **Leong Ting Yi**: Validation strategies, metrics, visualizations.
* **Hong Jia Xuan**: Abstract, introduction, literature review.

---

## 📚 Reference

* Chen, X., et al. *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675 (2023).
* He, K., et al. *Deep Residual Learning for Image Recognition*. CVPR (2016).


