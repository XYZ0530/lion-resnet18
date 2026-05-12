# ResNet-18 Optimizer Comparison on CIFAR-10

This project compares the Adam and Lion optimizers when training a ResNet-18 image classifier on CIFAR-10. The goal is to evaluate whether Lion can improve model accuracy and training efficiency under the same experimental setup.

## Project Overview

The project trains ResNet-18 models in PyTorch and compares optimizer performance across multiple runs. The final documented result shows that Lion achieved higher average accuracy and lower average training time than Adam on CIFAR-10.

Main project value:

- PyTorch deep learning experimentation
- Optimizer comparison and custom optimizer integration
- Model training, validation, and testing workflow
- Performance measurement across repeated runs
- Result documentation for technical communication

## Why Compare Adam and Lion?

Adam is a widely used adaptive optimizer and is a strong baseline for deep learning experiments. Lion is a newer sign-momentum optimizer that updates parameters using the sign of a momentum-based update. This project compares them in a controlled ResNet-18 training setup to observe differences in accuracy and training time.

## Dataset

- Dataset: CIFAR-10
- Task: Image classification
- Classes: 10 object categories
- Image size: 32 x 32 RGB images

CIFAR-10 is a standard benchmark dataset for small-scale computer vision experiments.

## Methodology

1. Train a ResNet-18 model on CIFAR-10 using Adam.
2. Train the same ResNet-18 architecture using Lion.
3. Keep the comparison focused on optimizer behavior.
4. Track accuracy, precision, recall, validation behavior, and training time.
5. Repeat the experiment across multiple runs and report average performance.

The repository includes the custom Lion optimizer implementation in `lion_optimizer.py` and notebooks for training and metric aggregation.

## Results

| Optimizer | Accuracy | Average Training Time |
|---|---:|---:|
| Adam | 78.40% | 403.37s |
| Lion | 81.16% | 270.62s |

Lion improved accuracy by **2.76 percentage points** and reduced average training time by **32.91%** compared with Adam.

## Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── lion_optimizer.py
├── Model_final.ipynb
├── Calculate_Average_Performance.ipynb
├── Train2.ipynb
├── Train3.ipynb
├── Train4.ipynb
└── Train5.ipynb
```

Key files:

- `lion_optimizer.py`: Custom PyTorch implementation of the Lion optimizer.
- `Train2.ipynb` to `Train5.ipynb`: Training notebooks for repeated optimizer comparison runs.
- `Model_final.ipynb`: Final model training and evaluation notebook.
- `Calculate_Average_Performance.ipynb`: Aggregates average accuracy, precision, recall, and training time.

## How to Run

Clone the repository:

```bash
git clone https://github.com/XYZ0530/lion-resnet18.git
cd lion-resnet18
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then run the notebooks in this order:

1. Open one of the training notebooks, such as `Model_final.ipynb` or a `Train*.ipynb` file.
2. Run `Calculate_Average_Performance.ipynb` to review the averaged metrics.

Note: CIFAR-10 may be downloaded automatically by torchvision when the notebook is run, depending on the dataset-loading cell.

## Limitations

- The experiment focuses only on CIFAR-10 and ResNet-18.
- The comparison is limited to Adam and Lion.
- Hyperparameter tuning is not presented as a broad optimizer search.
- Results should be interpreted as project-level experimental findings, not as a universal claim that Lion always outperforms Adam.

## What I Learned

- How to implement and integrate a custom optimizer in PyTorch.
- How optimizer choice affects training speed and model performance.
- How to structure repeated experiments for a fairer comparison.
- How to evaluate model results using accuracy, precision, recall, and training time.
- How to document deep learning experiments for technical and recruiter audiences.

## Resume Summary

Built a PyTorch ResNet-18 image classification experiment on CIFAR-10 comparing Adam and Lion optimizers. Lion achieved **81.16% accuracy** versus Adam's **78.40%**, while reducing average training time from **403.37s** to **270.62s** (**32.91% faster**), demonstrating deep learning experimentation, optimizer evaluation, and model performance analysis.
