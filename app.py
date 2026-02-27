import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for a premium feel
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f1f5f9;
    }
    
    /* Hide Streamlit components for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 2rem;
    }
    
    .sidebar-title {
        color: #38bdf8;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Cards Styling */
    .dashboard-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .dashboard-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.5);
        border-color: rgba(56, 189, 248, 0.4);
    }

    /* Header Styling */
    .main-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Metric Styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Input Styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    
    /* Result Styling */
    .result-success {
        color: #10b981;
        font-weight: 700;
    }
    
    .result-warning {
        color: #f59e0b;
        font-weight: 700;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #1d4ed8 100%);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.3);
        transform: scale(1.02);
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
with st.sidebar:
    st.markdown('<div class="sidebar-title">Candidate Profile</div>', unsafe_allow_html=True)
    
    city_dev = st.slider("City Development Index", 0.4, 1.0, 0.8, help="Scale of 0-1 reflecting city growth.")
    gender = st.selectbox("Gender", list(meta['gender_dict'].keys()))
    relevent_exp = st.selectbox("Relevant Experience", list(meta['experience_dict'].keys()))
    enrolled_univ = st.selectbox("Enrollment Type", list(meta['enrollment_dict'].keys()))
    edu_level = st.selectbox("Education Level", ["Graduate", "Masters", "Phd"])
    discipline = st.selectbox("Major Discipline", list(meta['discipline_dict'].keys()))
    
    exp_labels = ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"]
    experience = st.selectbox("Total Experience (Years)", exp_labels)
    
    comp_size_labels = ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"]
    company_size = st.selectbox("Company Size", comp_size_labels)
    
    company_type = st.selectbox("Current Company Type", list(meta['company_dict'].keys()))
    
    last_job_labels = ["1", "2", "3", "4", ">4", "never"]
    last_new_job = st.selectbox("Last Job Switch (Years)", last_job_labels)
    
    training_hours = st.number_input("Training Hours Completed", 1, 400, 50)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Profile")

# Map inputs to data
def process_input():
    data = {
        'city_development_index': city_dev,
        'gender': meta['gender_dict'][gender],
        'relevent_experience': meta['experience_dict'][relevent_exp],
        'enrolled_university': meta['enrollment_dict'][enrolled_univ],
        'education_level': meta['education_dict'][edu_level],
        'major_discipline': meta['discipline_dict'][discipline],
        'experience': exp_labels.index(experience), 
        'company_size': comp_size_labels.index(company_size),
        'company_type': meta['company_dict'][company_type],
        'last_new_job': last_job_labels.index(last_new_job),
        'training_hours': training_hours
    }
    return pd.DataFrame([data])

if model and meta:
    user_data = process_input()

    # Layout: Hero Section
    st.markdown('<h1 class="main-title">Job Change Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predicting career transitions with Machine Learning</p>', unsafe_allow_html=True)

    # Dashboard Row 1: Summary Stats
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f'<div class="dashboard-card"><p class="metric-label">Education</p><p class="metric-value">{edu_level}</p></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="dashboard-card"><p class="metric-label">Experience</p><p class="metric-value">{experience}y</p></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown(f'<div class="dashboard-card"><p class="metric-label">Training</p><p class="metric-value">{training_hours}h</p></div>', unsafe_allow_html=True)
    with m_col4:
        st.markdown(f'<div class="dashboard-card"><p class="metric-label">Discipline</p><p class="metric-value" style="font-size: 1.5rem;">{discipline}</p></div>', unsafe_allow_html=True)

    # Main Row: Prediction Result & Probability
    res_col, plot_col = st.columns([1, 1])
    
    if analyze_btn:
        with res_col:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            prediction = model.predict(user_data)[0]
            probability = model.predict_proba(user_data)[0][1]
            
            st.subheader("Prediction Analysis")
            if prediction == 1:
                st.markdown(f'### Status: <span class="result-warning">Active Job Hunter</span>', unsafe_allow_html=True)
                st.write("Candidate has a high statistical probability of seeking new opportunities.")
            else:
                st.markdown(f'### Status: <span class="result-success">Stable Employee</span>', unsafe_allow_html=True)
                st.write("Candidate is statistically likely to persist in their current role.")
            
            st.markdown(f"**Confidence Level**: `{probability*100:.1f}%` likelihood of change")
            st.progress(float(probability))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with res_col:
            st.markdown('<div class="dashboard-card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
            st.image("https://img.icons8.com/isometric/100/38bdf8/brain.png")
            st.markdown("### Ready for Analysis")
            st.write("Configure candidate attributes in the sidebar and click 'Analyze' to begin.")
            st.markdown('</div>', unsafe_allow_html=True)

    with plot_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Visual Profile")
        df_display = pd.read_csv("data/aug_train.csv").dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_display, x='target', y='city_development_index', palette='cool', ax=ax)
        ax.set_title("Target vs City Development Index", color='white')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bottom Row: Insights
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📈 Historical Trends")
    tab1, tab2 = st.tabs(["Training Intensity", "Major Distributions"])
    with tab1:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        sns.histplot(df_display['training_hours'], kde=True, color='#38bdf8', ax=ax2)
        fig2.patch.set_facecolor('#1e293b')
        ax2.set_facecolor('#1e293b')
        ax2.tick_params(colors='white')
        st.pyplot(fig2)
    with tab2:
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        sns.countplot(data=df_display, x='major_discipline', hue='target', palette='viridis', ax=ax3)
        fig3.patch.set_facecolor('#1e293b')
        ax3.set_facecolor('#1e293b')
        ax3.tick_params(colors='white')
        st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Please ensure the machine learning model is trained.")
