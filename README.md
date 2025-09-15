# Assignment 4 — GMM-Based Synthetic Sampling for Imbalanced Data  

**Name:** Priyanshu Verma  
**Roll no:** CH22B087  
**Date:** 15-09-2025  

---

## 📌 Project Overview  

This notebook demonstrates the application of a **Gaussian Mixture Model (GMM)-based synthetic sampling approach** for handling severe class imbalance in the context of **credit card fraud detection**.  

A baseline Logistic Regression classifier was first trained on the imbalanced dataset. To address the imbalance, the notebook applies a GMM to the minority (fraud) class to generate synthetic samples, combined with clustering-based undersampling of the majority class. This creates balanced training sets for further evaluation.  

The main deliverable is a Jupyter Notebook (`DA5401_Assignment_4.ipynb`) that walks through data preprocessing, GMM-based synthetic sampling, model training, and evaluation.

---

## 📂 Files in this Repository  

- `DA5401_Assignment_4.ipynb` — main notebook (completed assignment).  
- `README.md` — this file.  

---

## 📝 Notebook Structure  

The notebook is structured into the following logical sections:  

1. **Assignment Description** — Overview of the problem and objectives.  
2. **Part 1: Data Loading and Preprocessing** — Loading the dataset, scaling features, and splitting into train/test sets.  
3. **Part 2: Baseline Model** — Training Logistic Regression on the imbalanced dataset and evaluating performance.  
4. **Part 3: GMM-Based Synthetic Sampling** — Applying Gaussian Mixture Model (GMM) to generate synthetic samples of the minority class.  
5. **Part 4: Cluster-Based Undersampling** — Using KMeans clustering to undersample the majority class.  
6. **Part 5: Model Evaluation** — Training Logistic Regression on balanced data and comparing metrics.  
7. **Conclusion** — Final recommendation based on performance analysis.

---

## 📊 Results  

- **Baseline Logistic Regression Performance:**  
    - High precision (≈0.85)  
    - Strong F1-score (≈0.72)  

- **GMM-Based Sampling + Logistic Regression:**  
    - High recall (≈0.87)  
    - Extremely low precision and F1-score  

👉 The experimental results suggest that GMM-based synthetic sampling does not improve minority class detection when combined with Logistic Regression. The method leads to over-classifying non-fraudulent samples as fraudulent.  

**Final Recommendation:** GMM-based sampling should not be used with Logistic Regression for this task. More expressive models like Random Forest or Gradient Boosting may yield better results.

---

## ⚙️ Requirements  

Recommended Python environment:  

- Python 3.8+ (3.9/3.10 recommended)  
- JupyterLab or Jupyter Notebook (or Google Colab)  

### Python Packages  
Install the following packages:  

```bash
pip install pandas numpy scikit-learn seaborn
