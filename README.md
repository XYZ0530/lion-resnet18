# lion-resnet18
ResNet-18 on CIFAR-10 with LION optimizer; +2.7% acc vs Adam and ~33% faster training.


# Lion-ResNet18: Optimizing Image Classification with the Lion Optimizer

## 🦁 Background on Lion

The Lion optimizer (Evolved Sign Momentum) is a recently proposed deep learning optimizer developed by researchers at Google and UCLA (Chen et al., 2023). Instead of relying on adaptive second-order moments like Adam, Lion uses a sign-based update rule with momentum. This makes it memory-efficient, easier to tune with lower learning rates, and often capable of achieving faster convergence while maintaining strong generalization. Lion has shown excellent results on architectures like Vision Transformers and MLP-Mixers, but had not been tested on ResNet-18 before this project.


## 📌 Overview

This project evaluates the **Lion optimizer** on **ResNet-18** for CIFAR-10 image classification, comparing it against **Adam** under identical training conditions. Lion uses a **sign-based update** that tends to converge faster with lower memory overhead. In our experiments, Lion delivers higher accuracy and significantly shorter training time than Adam, averaged across multiple independent runs.&#x20;

---

## 🚀 Key Results

* **Accuracy (mean of 5 runs)**

  * ResNet-18 + **Lion** → **81.16%**
  * ResNet-18 + **Adam** → **78.40%**.&#x20;
* **Training time**

  * Lion averaged **270.62 s** vs Adam **403.37 s** (**\~32.91% faster**).&#x20;
* **Convergence behavior**

  * Lion rises faster on validation accuracy and maintains lower, steadier validation loss, reaching its peak early with stable early-stopping. &#x20;

---

## 🔬 Methodology

1. **Model & Dataset**

   * Architecture: **ResNet-18** (residual blocks with skip connections), chosen for its balance of capacity and efficiency. &#x20;
   * Dataset: **CIFAR-10** (50k train / 10k test, 32×32×3).&#x20;

2. **Optimizers & Hyperparameters**

   * **Adam** baseline with **LR = 0.001**.
   * **Lion** with **LR = 0.0001** (i.e., `0.001 / 10`) to stabilize the sign-based update; Lion generally benefits from a smaller LR.&#x20;
   * Lion implementation: **custom PyTorch optimizer** adapted from *lucidrains/lion-pytorch*.&#x20;

3. **Training Procedure**

   * **Early stopping**: `patience=3`, `min_delta=0.01` (monitoring validation loss).
   * **Five independent runs** per optimizer; we report averaged metrics for robustness.&#x20;

---

## 🧪 Exact Training Snippet (as used)

```python
# Early stopping parameters
early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=True)

# Train ResNet-18 (Adam)
print("Training Resnet-18(Adam)")
Resnet18_adam = ResNet18().to(device)
optimizer_adam = optim.Adam(Resnet18_adam.parameters(), lr=0.001)
train_losses_adam, train_accuracies_adam, val_losses_adam, val_accuracies_adam, train_time_adam = [], [], [], [], []
train_model_with_early_stopping(
    Resnet18_adam, optimizer_adam,
    train_losses_adam, train_accuracies_adam, val_losses_adam, val_accuracies_adam,
    early_stopping, train_time_adam
)
print("ResNet-18(Adam) training complete with early stopping\n")

# Reset early stopping
early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=True)

# Train ResNet-18 (Lion)
print("Training Resnet-18(Lion)")
Resnet18_lion = ResNet18().to(device)
optimizer_lion = Lion(Resnet18_lion.parameters(), lr=0.001/10)  # i.e., 0.0001
train_losses_lion, train_accuracies_lion, val_losses_lion, val_accuracies_lion, train_time_lion = [], [], [], [], []
train_model_with_early_stopping(
    Resnet18_lion, optimizer_lion,
    train_losses_lion, train_accuracies_lion, val_losses_lion, val_accuracies_lion,
    early_stopping, train_time_lion
)
print("ResNet-18(Lion) training complete with early stopping\n")
```

---

## 📈 What We Measure

* **Accuracy (test set)**, **precision**, **recall**
* **Validation curves** (loss & accuracy) to visualize convergence
* **Training time** to compare efficiency
* **Average across 5 runs** for statistical reliability (see final aggregation notebook)&#x20;

---

## 🛠️ Repository Structure

```
├── lion_optimizer.py                 # Custom PyTorch Lion optimizer (adapted from lucidrains/lion-pytorch)
├── Train2.ipynb                      # Training run (Adam vs Lion) with early stopping
├── Train3.ipynb
├── Train4.ipynb
├── Train5.ipynb
├── Model_final.ipynb                 # Aggregates metrics and plots
├── Calculate_Average_Performance.ipynb  # Computes averaged accuracy/time/precision/recall across runs
├── Report.pdf                        # Full write-up (methods, analysis, results)
└── README.md
```

---

## ⚙️ Setup

**Requirements**

* Python 3.8+
* PyTorch ≥ 1.12, torchvision
* numpy, matplotlib, seaborn (viz), scikit-learn (metrics)

**Install**

```bash
git clone https://github.com/<your-username>/lion-resnet18.git
cd lion-resnet18
pip install -r requirements.txt
```

**Run**

* Open any `TrainX.ipynb` to reproduce a single run (Adam vs Lion with early stopping).
* Open `Model_final.ipynb` and `Calculate_Average_Performance.ipynb` to view **mean metrics and charts** across runs.

---

## 🧠 How Lion Works (Intuition)

Lion uses **sign-momentum** updates with weight decay, simplifying state (no 2nd-moment buffers) and often yielding **faster, stabler convergence** with lower memory. On CIFAR-10 + ResNet-18, this manifested as **earlier accuracy gains** and **smoother loss** compared to Adam during training/validation.  &#x20;

---

## 👥 Contributions

* **Tan Yi Zhao** — ResNet-18 + Lion integration, training pipeline (early stopping), data preprocessing, average-metrics computation, report consolidation.
* **Sim Wen Ken** — Data prep, architecture description.
* **Leong Ting Yi** — Validation strategy, metrics, visualizations.
* **Hong Jia Xuan** — Abstract, introduction, literature review.

---

## 📚 References

* Chen et al., *Symbolic Discovery of Optimization Algorithms*, 2023.
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
* Project report excerpts used in this README: see `Report.pdf` (abstract, methods, results).&#x20;


