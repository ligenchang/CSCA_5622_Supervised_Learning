"""
Credit Approval Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Optional, Dict, Any

# Configuration
PAGE_CONFIG = {
    "page_title": "Credit Approval Prediction System",
    "page_icon": "",
    "layout": "wide"
}

MODEL_PATHS = {
    "model": "models/credit_approval_model.pkl",
    "features": "models/feature_names.pkl", 
    "metadata": "models/model_metadata.pkl"
}

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Model Loading Functions
@st.cache_resource
def load_model_components() -> Tuple[Any, Any, Dict[str, Any], bool]:
    """
    Load the trained model and preprocessing components
    
    Returns:
        Tuple containing (model, feature_names, metadata, success_flag)
    """
    try:
        model = joblib.load(MODEL_PATHS["model"])
        feature_names = joblib.load(MODEL_PATHS["features"])
        metadata = joblib.load(MODEL_PATHS["metadata"])
        
        return model, feature_names, metadata, True
        
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None, False
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None, False

# Initialize model components
model, feature_names, model_metadata, model_loaded = load_model_components()

# UI Components
def render_header():
    """Render the main page header and description"""
    st.title("Credit Approval Prediction System")

def render_sidebar():
    """Render the sidebar with model information"""
    st.sidebar.header("Model Information")
    
    if model_loaded:
        st.sidebar.success("Model loaded successfully!")
        
        # Performance metrics
        metrics = {
            'Accuracy': model_metadata.get('accuracy', 0),
            'Precision': model_metadata.get('precision', 0),
            'Recall': model_metadata.get('recall', 0),
            'F1-Score': model_metadata.get('f1_score', 0),
            'AUC-ROC': model_metadata.get('auc_roc', 0)
        }
        
        metrics_text = "\n".join([
            f"- {k}: {v:.1%}" if k != 'AUC-ROC' else f"- {k}: {v:.3f}"
            for k, v in metrics.items()
        ])
        
        st.sidebar.info(f"""
**Performance Metrics:**
{metrics_text}

**Model Type:** {model_metadata.get('model_type', 'XGBoost')}
**Features:** {len(feature_names)} processed attributes
**Dataset:** UCI Credit Approval
""")
    else:
        st.sidebar.error("Model files not found!")
        st.sidebar.warning("Please ensure model files are in the same directory.")

# Render UI components
render_header()
render_sidebar()

# Prediction Functions
def get_model_features() -> list:
    """Return the list of features expected by the model"""
    return [
        'A2', 'A3', 'A8', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15',  # Numerical features
        'A1_a', 'A1_b',  # A1 (sex) one-hot encoded
        'A4_l', 'A4_u', 'A4_y',  # A4 (employment_status) one-hot encoded
        'A5_g', 'A5_gg', 'A5_p',  # A5 (credit_purpose) one-hot encoded
        'A6_aa', 'A6_c', 'A6_cc', 'A6_d', 'A6_e', 'A6_ff', 'A6_i', 'A6_j',
        'A6_k', 'A6_m', 'A6_q', 'A6_r', 'A6_w', 'A6_x',  # A6 (job_category) one-hot encoded
        'A7_bb', 'A7_dd', 'A7_ff', 'A7_h', 'A7_j', 'A7_n', 'A7_o', 'A7_v', 'A7_z',  # A7 (housing_status)
        'A13_g', 'A13_p', 'A13_s'  # A13 (guarantor_status) one-hot encoded
    ]

def map_numerical_features(model_input: pd.DataFrame, data: Dict[str, Any]) -> None:
    """Map numerical features from user input to model format"""
    numerical_mappings = {
        'A2': 'age',
        'A3': 'credit_amount',
        'A8': 'monthly_income',
        'A11': 'employment_duration',
        'A14': 'credit_duration_months',
        'A15': 'existing_credits'
    }
    
    for model_feature, data_key in numerical_mappings.items():
        model_input[model_feature] = float(data[data_key])

def map_binary_features(model_input: pd.DataFrame, data: Dict[str, Any]) -> None:
    """Map binary features from user input to model format"""
    binary_mappings = {
        'A9': 'credit_history_flag',
        'A10': 'has_checking_account',
        'A12': 'property_ownership'
    }
    
    for model_feature, data_key in binary_mappings.items():
        model_input[model_feature] = 1.0 if data[data_key] == 't' else 0.0

def map_categorical_features(model_input: pd.DataFrame, data: Dict[str, Any]) -> None:
    """Map categorical features (one-hot encoded) from user input to model format"""
    categorical_mappings = {
        'sex': 'A1',
        'employment_status': 'A4',
        'credit_purpose': 'A5',
        'job_category': 'A6',
        'housing_status': 'A7',
        'guarantor_status': 'A13'
    }
    
    for data_key, model_prefix in categorical_mappings.items():
        model_input[f"{model_prefix}_{data[data_key]}"] = 1.0

def predict_credit_approval(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Predict credit approval using the trained XGBoost model
    
    Args:
        data: Dictionary containing user input data
        
    Returns:
        Tuple of (decision, confidence) or (None, None) for denied applications
    """
    if not model_loaded:
        return "MODEL NOT LOADED", 0.0
    
    try:
        # Initialize model input DataFrame
        model_features = get_model_features()
        model_input = pd.DataFrame(0, index=[0], columns=model_features, dtype=float)
        
        # Map features from user input to model format
        map_numerical_features(model_input, data)
        map_binary_features(model_input, data)
        map_categorical_features(model_input, data)
        
        # Make prediction
        prediction = model.predict(model_input)[0]
        prediction_proba = model.predict_proba(model_input)[0]
        confidence = max(prediction_proba)
        
        # Return results as APPROVED or REJECTED
        if prediction == 1:
            return "APPROVED", confidence
        else:
            return "REJECTED", confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "PREDICTION ERROR", 0.0

# Input Form Functions
def render_personal_info_inputs() -> Dict[str, Any]:
    """Render personal information input fields"""
    st.subheader("Personal Information")
    
    return {
        'sex': st.selectbox("A1 - Sex/Gender", ['a', 'b']),
        'age': st.number_input("A2 - Age", min_value=18.0, max_value=80.0, value=35.0),
        'employment_status': st.selectbox("A4 - Employment Status", ['u', 'y', 'l']),
        'employment_duration': st.number_input("A11 - Employment Duration (Years)", min_value=0, max_value=40, value=5),
        'job_category': st.selectbox("A6 - Job Category", ['w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff', 'j'])
    }

def render_financial_info_inputs() -> Dict[str, Any]:
    """Render financial information input fields"""
    st.subheader("Financial Information")
    
    return {
        'monthly_income': st.number_input("A8 - Monthly Income (Thousands)", min_value=0.0, max_value=50.0, value=3.0, step=0.1),
        'credit_amount': st.number_input("A3 - Credit Amount (Thousands)", min_value=0.1, max_value=100.0, value=10.0, step=0.1),
        'existing_credits': st.number_input("A15 - Existing Credits", min_value=0.0, max_value=10000.0, value=500.0),
        'credit_duration_months': st.number_input("A14 - Credit Duration (Months)", min_value=1.0, max_value=360.0, value=36.0),
        'credit_purpose': st.selectbox("A5 - Credit Purpose", ['g', 'p', 'gg'])
    }

def render_account_property_inputs() -> Dict[str, Any]:
    """Render account and property status input fields"""
    st.subheader("Account & Property Status")
    
    return {
        'credit_history_flag': st.selectbox("A9 - Credit History", ['t', 'f'], format_func=lambda x: 'Good' if x == 't' else 'Poor'),
        'has_checking_account': st.selectbox("A10 - Has Checking Account", ['t', 'f'], format_func=lambda x: 'Yes' if x == 't' else 'No'),
        'property_ownership': st.selectbox("A12 - Property Ownership", ['t', 'f'], format_func=lambda x: 'Yes' if x == 't' else 'No'),
        'housing_status': st.selectbox("A7 - Housing Status", ['v', 'h', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n']),
        'guarantor_status': st.selectbox("A13 - Guarantor Status", ['g', 's', 'p'])
    }

def collect_user_input() -> Dict[str, Any]:
    """Collect all user input from the form"""
    
    # Restore previous 3-column format, but order fields by A1, A2, ...
    col1, col2, col3 = st.columns(3)
    with st.form("credit_app_form"):
        with col1:
            sex = st.selectbox("A1 - Sex/Gender", ['a', 'b'])
            age = st.number_input("A2 - Age", value=35.0)
            credit_amount = st.number_input("A3 - Credit Amount (Thousands)", value=10.0, step=0.1)
            employment_status = st.selectbox("A4 - Employment Status", ['u', 'y', 'l'])
            credit_purpose = st.selectbox("A5 - Credit Purpose", ['g', 'p', 'gg'])
        with col2:
            job_category = st.selectbox("A6 - Job Category", ['w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff', 'j'])
            housing_status = st.selectbox("A7 - Housing Status", ['v', 'h', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n'])
            monthly_income = st.number_input("A8 - Monthly Income (Thousands)", value=3.0, step=0.1)
            credit_history_flag = st.selectbox("A9 - Credit History", ['t', 'f'], format_func=lambda x: 'Good' if x == 't' else 'Poor')
            has_checking_account = st.selectbox("A10 - Has Checking Account", ['t', 'f'], format_func=lambda x: 'Yes' if x == 't' else 'No')
        with col3:
            employment_duration = st.number_input("A11 - Employment Duration (Years)", value=5)
            property_ownership = st.selectbox("A12 - Property Ownership", ['t', 'f'], format_func=lambda x: 'Yes' if x == 't' else 'No')
            guarantor_status = st.selectbox("A13 - Guarantor Status", ['g', 's', 'p'])
            credit_duration_months = st.number_input("A14 - Credit Duration (Months)", value=36.0)
            existing_credits = st.number_input("A15 - Existing Credits", value=500.0)
        submitted = st.form_submit_button("Analyze Credit Application")
    return {
        'sex': sex,
        'age': age,
        'credit_amount': credit_amount,
        'employment_status': employment_status,
        'credit_purpose': credit_purpose,
        'job_category': job_category,
        'housing_status': housing_status,
        'monthly_income': monthly_income,
        'credit_history_flag': credit_history_flag,
        'has_checking_account': has_checking_account,
        'employment_duration': employment_duration,
        'property_ownership': property_ownership,
        'guarantor_status': guarantor_status,
        'credit_duration_months': credit_duration_months,
        'existing_credits': existing_credits,
        'form_submitted': submitted
    }

# Main Application Functions
def display_prediction_results(decision: str, confidence: float) -> None:
    """Display prediction results in a formatted layout"""
    col1, col2 = st.columns(2)
    if decision == "APPROVED":
        decision_html = f'<span style="color:green;font-weight:bold;font-size:1.5em;">APPROVED</span>'
    elif decision == "REJECTED":
        decision_html = f'<span style="color:red;font-weight:bold;font-size:1.5em;">REJECTED</span>'
    else:
        decision_html = f'<span style="font-weight:bold;font-size:1.5em;">{decision}</span>'
    with col1:
        st.markdown(f"**Decision**<br>{decision_html}", unsafe_allow_html=True)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

def run_prediction_workflow() -> None:
    """Run the main prediction workflow"""    
    # Collect user input
    user_data = collect_user_input()
    if not user_data.get('form_submitted'):
        return
    # Remove form_submitted key
    user_data = {k: v for k, v in user_data.items() if k != 'form_submitted'}
    if not model_loaded:
        st.error("Cannot make prediction - model not loaded!")
        st.info("Please ensure all model files are available in the application directory.")
        return
    # Make prediction
    decision, confidence = predict_credit_approval(user_data)
    # Display results as APPROVED or REJECTED
    if decision == "APPROVED":
        display_prediction_results("APPROVED", confidence)
    elif decision == "REJECTED":
        display_prediction_results("REJECTED", confidence)
    elif decision == "PREDICTION ERROR":
        st.error("An error occurred during prediction. Please try again.")
    else:
        st.info("Application could not be processed. Please adjust parameters and try again.")

# Run the main application
run_prediction_workflow()