import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score, roc_curve)
import pickle

# Set page config
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# Title
st.title("Breast Cancer Classification App")
st.write("""
This app uses a Logistic Regression model to classify breast cancer tumors as malignant or benign.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application was developed to demonstrate a machine learning model for breast cancer classification.
The model was trained on the Wisconsin Breast Cancer Dataset.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\vasanth\Downloads\breast_cancer_data.csv")
    return df

df = load_data()

# Display raw data
if st.checkbox("Show raw data"):
    st.subheader("Breast Cancer Dataset")
    st.write(df.head())

# Display data info
if st.checkbox("Show data information"):
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# EDA Visualizations
st.subheader("Exploratory Data Analysis")

# Target distribution
st.write("### Target Class Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=df, x='target', ax=ax)
ax.set_xlabel('Target (0 = Malignant, 1 = Benign)')
ax.set_ylabel('Count')
st.pyplot(fig)

# Correlation heatmap
st.write("### Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Model training and evaluation
st.subheader("Model Training and Evaluation")

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=45)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Display metrics
st.write("### Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

with col2:
    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# ROC Curve
st.write("### ROC Curve")
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)

# Prediction interface
st.subheader("Make a Prediction")

# Create input fields for features
st.write("Enter tumor characteristics to make a prediction:")

# Group features into columns
col1, col2, col3 = st.columns(3)

# Mean features
with col1:
    st.write("**Mean Values**")
    mean_radius = st.number_input("Mean Radius", min_value=0.0, value=14.0, step=0.1)
    mean_texture = st.number_input("Mean Texture", min_value=0.0, value=19.0, step=0.1)
    mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, value=92.0, step=0.1)
    mean_area = st.number_input("Mean Area", min_value=0.0, value=655.0, step=1.0)
    mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, value=0.1, step=0.01)

with col2:
    st.write("**Worst Values**")
    worst_radius = st.number_input("Worst Radius", min_value=0.0, value=16.0, step=0.1)
    worst_texture = st.number_input("Worst Texture", min_value=0.0, value=25.0, step=0.1)
    worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, value=107.0, step=0.1)
    worst_area = st.number_input("Worst Area", min_value=0.0, value=880.0, step=1.0)
    worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, value=0.13, step=0.01)

with col3:
    st.write("**Other Features**")
    mean_compactness = st.number_input("Mean Compactness", min_value=0.0, value=0.1, step=0.01)
    mean_concavity = st.number_input("Mean Concavity", min_value=0.0, value=0.09, step=0.01)
    mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, value=0.05, step=0.01)
    worst_compactness = st.number_input("Worst Compactness", min_value=0.0, value=0.25, step=0.01)
    worst_concavity = st.number_input("Worst Concavity", min_value=0.0, value=0.27, step=0.01)

# Create a dictionary with all features (including those not shown in the UI)
input_data = {
    'mean radius': mean_radius,
    'mean texture': mean_texture,
    'mean perimeter': mean_perimeter,
    'mean area': mean_area,
    'mean smoothness': mean_smoothness,
    'mean compactness': mean_compactness,
    'mean concavity': mean_concavity,
    'mean concave points': mean_concave_points,
    'mean symmetry': 0.18,  # default values for features not in UI
    'mean fractal dimension': 0.06,
    'radius error': 0.41,
    'texture error': 1.22,
    'perimeter error': 2.87,
    'area error': 40.0,
    'smoothness error': 0.01,
    'compactness error': 0.03,
    'concavity error': 0.03,
    'concave points error': 0.01,
    'symmetry error': 0.02,
    'fractal dimension error': 0.003,
    'worst radius': worst_radius,
    'worst texture': worst_texture,
    'worst perimeter': worst_perimeter,
    'worst area': worst_area,
    'worst smoothness': worst_smoothness,
    'worst compactness': worst_compactness,
    'worst concavity': worst_concavity,
    'worst concave points': 0.11,
    'worst symmetry': 0.29,
    'worst fractal dimension': 0.08
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale the input
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction_prob = model.predict_proba(input_scaled)
    
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.success("The tumor is predicted to be **Benign** (non-cancerous)")
    else:
        st.error("The tumor is predicted to be **Malignant** (cancerous)")
    
    st.write(f"Probability of being Benign: {prediction_prob[0][1]:.2%}")
    st.write(f"Probability of being Malignant: {prediction_prob[0][0]:.2%}")

# Download model
st.sidebar.header("Model Download")
st.sidebar.download_button(
    label="Download Trained Model",
    data=pickle.dumps(model),
    file_name="breast_cancer_model.pkl",
    mime="application/octet-stream"
)
