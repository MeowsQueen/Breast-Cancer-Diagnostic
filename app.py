import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Breast Cancer Prediction App")
st.write("Evaluate and visualize machine learning models for breast cancer prediction.")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    df_original = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df_original.head())

    # Check for necessary columns
    if 'diagnosis' not in df_original.columns:
        st.error("The uploaded dataset does not contain the 'diagnosis' column!")
        st.stop()

    # Data preprocessing
    df_original.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')
    label_encoder = LabelEncoder()
    df_original['diagnosis'] = label_encoder.fit_transform(df_original['diagnosis'])  # 0: Benign, 1: Malignant

    x = df_original.drop(columns=['diagnosis'])
    y = df_original['diagnosis']

    # Scaling features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=x.columns)

    # Class balancing
    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    x = pd.DataFrame(x_resampled, columns=x.columns)
    y = pd.Series(y_resampled)
    st.write("Class balancing applied.")

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    if st.checkbox("Show summary statistics"):
        st.write(df_original.describe())

    if st.checkbox("Show correlation matrix"):
        corr = df_original.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Model selection and training
    model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Support Vector Machine (SVM)", "Gradient Boosting Machine (GBM)"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42)
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

    with st.spinner("Performing hyperparameter tuning..."):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    st.write(f"Best hyperparameters: {grid_search.best_params_}")

    # Model evaluation
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    aucroc = roc_auc_score(y_test, y_pred_proba)

    st.write("### Model Performance Metrics")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**AUC-ROC:** {aucroc:.4f}")

    # Prediction vs Actual Visualization
    st.subheader("Prediction vs Actual Data")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    ax.set_xlabel("Actual Diagnosis")
    ax.set_ylabel("Predicted Diagnosis")
    ax.set_title("Prediction vs Actual")
    st.pyplot(fig)

    # Feature Importance
    if model_choice == "Gradient Boosting Machine (GBM)" and st.checkbox("Show Feature Importance"):
        feature_importance = pd.Series(best_model.feature_importances_, index=x.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)
