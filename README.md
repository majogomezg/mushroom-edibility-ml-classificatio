# 🍄 Mushroom Edibility Classification – Machine Learning Pipeline

This project builds an end-to-end machine learning pipeline to classify mushrooms as **edible** or **poisonous** using a benchmark dataset from Kaggle.  
The solution evaluates **classical machine learning algorithms**—Support Vector Machines (SVM), Multilayer Perceptrons (MLP), and an ID3-like Decision Tree—under a structured experimental design.

All preprocessing, training, inference and evaluation steps are implemented using **scikit-learn**.

---

## 💡 Project Objectives

- Build a reproducible machine learning workflow for a binary classification task.
- Apply classical ML techniques (no deep learning).
- Compare performance across multiple model configurations.
- Evaluate models using robust metrics beyond accuracy, such as Precision, Recall, F1-score, ROC-AUC and PR-AUC.
- Generate insights about which algorithms perform best for categorical, high-cardinality data.

---

## 📦 Dataset

**Source:** Kaggle → `devzohaib/mushroom-edibility-classification`  
**Format:** CSV  
**Target:**  
- `e` = edible  
- `p` = poisonous  

**Features:**  
- 21 categorical attributes (cap shape, odor, gill size, spore print color, etc.)
- No numerical variables — pipeline uses `OneHotEncoder`.

---

## 🧹 Preprocessing

- Automatic CSV separator detection  
- Replacement of unknown values (`? → "unknown"`)  
- Full One-Hot Encoding of categorical features  
- Output encoded as `float32` to reduce memory usage and improve computation time  
- Stratified split (80% train, 20% test)

---

## 🤖 Models Evaluated

### **1. Support Vector Machines (SVM)**  
- `LinearSVC` (fast linear classifier)  
- `SVC` with RBF kernel (`C=1`, `C=10`)  

### **2. Multilayer Perceptron (MLP)**  
All with **early stopping**:
- 1 hidden layer (50 units, ReLU)  
- 2 hidden layers (50, 20 units, ReLU)  
- 1 hidden layer (50 units, Tanh)

### **3. ID3-like Decision Tree**  
- `criterion="entropy"` to simulate ID3 behavior  
- Unlimited depth (controlled by entropy split)

---

## 📊 Evaluation Metrics

Each model is evaluated using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | How many predicted poisonous mushrooms were actually poisonous |
| **Recall** | How many poisonous mushrooms were detected correctly |
| **F1-score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Measures ranking performance (true positive rate vs false positive rate) |
| **PR-AUC** | Measures performance under class imbalance (precision vs recall curve) |
| **Confusion Matrix** | Distribution of TP, TN, FP, FN |

