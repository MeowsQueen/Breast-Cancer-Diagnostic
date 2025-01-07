import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Breast Cancer Classification App 🧬")
st.write("Unlock the power of ML to classify breast tumors as **Malignant** or **Benign** with **SVM**, **Gradient Boosting**, and **Logistic Regression**. Classification backed by me as a **Molecular Biologist**!")

# Image Display
st.image("https://www.frontiersin.org/files/Articles/629321/fonc-11-629321-HTML-r1/image_m/fonc-11-629321-g001.jpg", 
         caption="Examples of Benign and Malignant Masses on Mammograms 🩺", use_container_width=True)

# Load Dataset
df = pd.read_csv('data.csv')
df_original = df.copy()
st.write("Dataset Preview 📊")
st.dataframe(df.head())

# Data Preprocessing
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])  # 0: Benign, 1: Malignant

# Split Features and Target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Class Balancing with RUS
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Model selection
model_choice = st.selectbox("Choose a model 🔍:", ["Logistic Regression", "Support Vector Machine (SVM)", "Gradient Boosting Machine (GBM)"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
elif model_choice == "Support Vector Machine (SVM)":
    model = SVC(probability=True, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
elif model_choice == "Gradient Boosting Machine (GBM)":
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'min_samples_split': [2, 5]
    }

# Hyperparameter Tuning
with st.spinner("Performing hyperparameter tuning... ⏳"):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
st.write(f"Best hyperparameters: {grid_search.best_params_} 🔧")

# Model Evaluation
y_pred = best_model.predict(X_test)
if hasattr(best_model, 'predict_proba'):
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
else:
    y_pred_proba = best_model.decision_function(X_test)
    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())  # Normalize decision function

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
aucroc = roc_auc_score(y_test, y_pred_proba)

# Display Metrics
st.write("### Model Performance Metrics 📈")
st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")
st.write(f"**AUC-ROC:** {aucroc:.4f}")

# Visualization Options
st.subheader("Visualization Options 📊")

# Correlation Matrix
if st.checkbox("Show Correlation Matrix 🔗"):
    st.subheader("Correlation Matrix 🔗")
    corr = df_original.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

# ROC Curve
if st.checkbox("Show ROC Curve 📉"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Confusion Matrix
if st.checkbox("Show Confusion Matrix 🔲"):
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

# Feature Importance
if st.checkbox("Show Feature Importance 💡"):
    st.subheader("Feature Importance 💡")
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)
    elif hasattr(best_model, 'coef_'):
        feature_importance = pd.Series(best_model.coef_[0], index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)
    else:
        st.write("Feature Importance is not available for this model.")
