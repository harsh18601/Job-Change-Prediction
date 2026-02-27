import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for a premium feel
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    /* Styling for the main container */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1 {
        color: #38bdf8;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Card/Container styling */
    .prediction-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(56, 189, 248, 0.4);
    }
    
    /* Input border customization */
    .stSelectbox, .stNumberInput {
        background: #334155;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model and Metadata
@st.cache_resource
def load_assets():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        return model, meta
    except FileNotFoundError:
        st.error("Model or Metadata files not found. Please run train_model.py first.")
        return None, None

model, meta = load_assets()

# Sidebar - User Inputs
st.sidebar.image("https://img.icons8.com/bubbles/100/000000/business-group.png", width=100)
st.sidebar.title("Candidate Info")
st.sidebar.markdown("---")

def get_user_input():
    city_dev = st.sidebar.slider("City Development Index", 0.4, 1.0, 0.8)
    gender = st.sidebar.selectbox("Gender", list(meta['gender_dict'].keys()))
    relevent_exp = st.sidebar.selectbox("Relevant Experience", list(meta['experience_dict'].keys()))
    enrolled_univ = st.sidebar.selectbox("Enrollment Type", list(meta['enrollment_dict'].keys()))
    edu_level = st.sidebar.selectbox("Education Level", ["Graduate", "Masters", "Phd"]) # Simplified keys
    discipline = st.sidebar.selectbox("Major Discipline", list(meta['discipline_dict'].keys()))
    
    # We need to map back to the encoding used in training for the rest
    exp_labels = ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"]
    experience = st.sidebar.selectbox("Total Experience (Years)", exp_labels)
    
    comp_size_labels = ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"]
    company_size = st.sidebar.selectbox("Company Size", comp_size_labels)
    
    company_type = st.sidebar.selectbox("Company Type", list(meta['company_dict'].keys()))
    
    last_job_labels = ["1", "2", "3", "4", ">4", "never"]
    last_new_job = st.sidebar.selectbox("Last Job Switch (Years)", last_job_labels)
    
    training_hours = st.sidebar.number_input("Training Hours", 1, 400, 50)
    
    # Map back to encoded values
    data = {
        'city_development_index': city_dev,
        'gender': meta['gender_dict'][gender],
        'relevent_experience': meta['experience_dict'][relevent_exp],
        'enrolled_university': meta['enrollment_dict'][enrolled_univ],
        'education_level': meta['education_dict'][edu_level],
        'major_discipline': meta['discipline_dict'][discipline],
        # For the LabelEncoded fields, we use the values we have but need to fix training script to be more robust
        'experience': exp_labels.index(experience), 
        'company_size': comp_size_labels.index(company_size),
        'company_type': meta['company_dict'][company_type],
        'last_new_job': last_job_labels.index(last_new_job),
        'training_hours': training_hours
    }
    return pd.DataFrame([data])

if model and meta:
    user_data = get_user_input()

    # Main Page
    st.title("🚀 Job Change Prediction")
    st.markdown("Predict how likely a candidate is to look for a new job based on their professional profile.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if st.button("Analyze Profile"):
            prediction = model.predict(user_data)[0]
            probability = model.predict_proba(user_data)[0][1]
            
            st.markdown("### Analysis Result")
            if prediction == 1:
                st.markdown(f"#### 🟠 Result: Likely to Change Jobs")
                st.progress(float(probability))
                st.write(f"Confidence Level: **{probability*100:.2f}%**")
            else:
                st.markdown(f"#### 🟢 Result: Likely to Stay")
                st.progress(float(1 - probability))
                st.write(f"Confidence Level: **{(1-probability)*100:.2f}%**")
                
            st.info("💡 Insights: Candidates with more training hours and PhD/Masters often show different patterns in job stability.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="prediction-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### Profile Summary")
        st.write(f"🎓 **Education**: {meta['columns'][5] if 'columns' in meta else 'Detail'}")
        st.write(f"💼 **Experience**: {user_data['experience'].values[0]} units")
        st.write(f"⌛ **Training**: {user_data['training_hours'].values[0]} hrs")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualization Section
    st.markdown("---")
    st.subheader("📊 Dataset Insights")
    
    tabs = st.tabs(["Overview", "Training Hours", "Exp Distribution"])
    
    with tabs[0]:
        st.write("Distribution of candidates likely to change vs stay.")
        df_display = pd.read_csv("data/aug_train.csv").dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df_display, x='target', palette='viridis', ax=ax)
        ax.set_title("Target Distribution in Training Data")
        st.pyplot(fig)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_display['training_hours'], kde=True, color='#38bdf8', ax=ax)
        ax.set_title("Training Hours Distribution")
        st.pyplot(fig)

    with tabs[2]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_display, x='target', y='city_development_index', palette='magma', ax=ax)
        ax.set_title("City Development Index vs Job Change")
        st.pyplot(fig)

else:
    st.warning("Please ensure train_model.py has been run successfully.")
