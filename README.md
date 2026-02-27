# 💼 Job Change Prediction Web App

This project predicts the likelihood of candidates looking for a new job based on their professional profiles. It uses an **XGBoost Classifier** and provides a professional web interface built with **Streamlit**.

## 🚀 Features
- **Modern UI**: Dark-themed, responsive interface with glassmorphism aesthetics.
- **Real-time Prediction**: Instantly analyze candidate profiles.
- **Data Insights**: Interactive visualizations showing dataset distributions.

## 📁 Project Structure
- `data/`: Contains the training and testing datasets.
- `notebooks/`: Original research and analysis notebooks.
- `train_model.py`: Script to preprocess data and train the model.
- `app.py`: Streamlit web application.
- `requirements.txt`: Project dependencies.

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Job-Change-Prediction.git
cd Job-Change-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Run the application
```bash
streamlit run app.py
```

## 📊 Dataset
The dataset includes attributes like city development index, gender, relevant experience, education level, and more. It is used to predict if a candidate is looking for a job (Target = 1) or not (Target = 0).
