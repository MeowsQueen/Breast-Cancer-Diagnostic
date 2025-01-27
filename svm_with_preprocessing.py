# -*- coding: utf-8 -*-
"""SVM WITH Preprocessing

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-zCXcUmJfqzg3ww5igTdoh4J4Nu7WCEF
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

df_original = pd.read_csv('/content/drive/MyDrive/ML PROJECT/data.csv')

# Preprocessing
df_original.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')

label_encoder = LabelEncoder()
df_original['diagnosis'] = label_encoder.fit_transform(df_original['diagnosis'])  # 0: Benign, 1: Malignant

x = df_original.drop(columns=['diagnosis'])
y = df_original['diagnosis']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)

# Balancing classes using Random Under Sampling
rus = RandomUnderSampler(random_state=42)
x_resampled, y_resampled = rus.fit_resample(x, y)

# Removing highly correlated features
def remove_highly_correlated_features(df, threshold=0.9):
    correlation_matrix = df.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]
    return df.drop(columns=to_drop), to_drop

x_resampled_df = pd.DataFrame(x_resampled, columns=x.columns)
x_resampled_cleaned, dropped_features = remove_highly_correlated_features(x_resampled_df, threshold=0.9)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled_cleaned, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define Stratified K-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# SVM model before tuning
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Cross-validation for baseline SVM model
cv_results = {
    "accuracy": cross_val_score(svm_model, x_train, y_train, cv=cv, scoring='accuracy'),
    "precision": cross_val_score(svm_model, x_train, y_train, cv=cv, scoring='precision'),
    "recall": cross_val_score(svm_model, x_train, y_train, cv=cv, scoring='recall'),
    "f1": cross_val_score(svm_model, x_train, y_train, cv=cv, scoring='f1'),
    "roc_auc": cross_val_score(svm_model, x_train, y_train, cv=cv, scoring='roc_auc'),
}

print("\n📊 Cross-Validation Results (Baseline SVM, mean ± std):")
for metric, scores in cv_results.items():
    print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

# Hyperparameter tuning with Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

# Best model after tuning
best_svm_model = grid_search.best_estimator_

# Evaluate baseline SVM model on test set
svm_model.fit(x_train, y_train)
y_pred_baseline = svm_model.predict(x_test)
baseline_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_baseline),
    "precision": precision_score(y_test, y_pred_baseline),
    "recall": recall_score(y_test, y_pred_baseline),
    "f1": f1_score(y_test, y_pred_baseline),
    "roc_auc": roc_auc_score(y_test, svm_model.predict_proba(x_test)[:, 1]),
}

print("\n📊 Final Model Evaluation (Baseline SVM, Test Set):")
for metric, score in baseline_metrics.items():
    print(f"{metric.capitalize()}: {score:.4f}")

# Evaluate best SVM model on test set
best_svm_model.fit(x_train, y_train)
y_pred_best = best_svm_model.predict(x_test)
best_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_best),
    "precision": precision_score(y_test, y_pred_best),
    "recall": recall_score(y_test, y_pred_best),
    "f1": f1_score(y_test, y_pred_best),
    "roc_auc": roc_auc_score(y_test, best_svm_model.predict_proba(x_test)[:, 1]),
}

print("\n📊 Best Model Evaluation (After Tuning, Test Set):")
for metric, score in best_metrics.items():
    print(f"{metric.capitalize()}: {score:.4f}")

# Confusion Matrix for Best Model
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve
y_pred_proba_best = best_svm_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"SVM Model (AUC = {auc(fpr, tpr):.4f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid(alpha=0.5)
plt.show()

# Combine metrics for visualization
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Baseline': list(baseline_metrics.values()),
    'Tuned': list(best_metrics.values())
})

# Bar plot to compare baseline and tuned metrics
metrics_df.set_index('Metric', inplace=True)
metrics_df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
plt.title('Baseline vs Tuned Model Performance (Test Set)')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(title='Model', loc='lower right')
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.show()