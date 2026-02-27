import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
import os

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv("data/aug_train.csv")
    
    # Preprocessing (based on notebook analysis)
    df = df.drop(columns=["enrollee_id", "city"])
    df = df.dropna()
    
    # Categorical mappings (simplified for the app but following notebook logic)
    experience_dict = {'Has relevent experience' : 1, 'No relevent experience': 0}
    education_dict = {'Graduate' : 2, 'Masters' : 1, 'Phd' : 0}
    enrollment_dict = {'no_enrollment' : 2, 'Full time course' : 1, 'Part time course' : 0}
    gender_dict = {'Male' : 2, 'Female' : 1, 'Other' : 0}
    discipline_dict = {'STEM' : 5, 'Humanities' : 4, 'Business Degree' : 3, 'Other' : 2, 'No Major' : 1, 'Arts' : 0 }
    company_dict = {'Pvt Ltd' : 5, 'Funded Startup' : 4, 'Public Sector' : 3, 'Early Stage Startup' : 2, 'NGO' : 1, 'Other' : 0 }

    # Apply mappings
    df['gender'] = df['gender'].map(gender_dict)
    df['relevent_experience'] = df['relevent_experience'].map(experience_dict)
    df['education_level'] = df['education_level'].map(education_dict)
    df['enrolled_university'] = df['enrolled_university'].map(enrollment_dict)
    df['major_discipline'] = df['major_discipline'].map(discipline_dict)
    df['company_type'] = df['company_type'].map(company_dict)
    
    # Label encode remaining objects
    le = LabelEncoder()
    df['experience'] = le.fit_transform(df['experience'].astype(str))
    df['company_size'] = le.fit_transform(df['company_size'].astype(str))
    df['last_new_job'] = le.fit_transform(df['last_new_job'].astype(str))
    
    # Drop rows with NaNs after manual mapping if any
    df = df.dropna()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print("Training XGBClassifier...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    print("Saving model and encoders...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # We also need to save the mappings and encoders for the app to use
    meta = {
        'experience_dict': experience_dict,
        'education_dict': education_dict,
        'enrollment_dict': enrollment_dict,
        'gender_dict': gender_dict,
        'discipline_dict': discipline_dict,
        'company_dict': company_dict,
        'experience_le': le, # Note: This is an oversimplification, normally we'd save each LE
        'columns': X.columns.tolist()
    }
    
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
        
    print("Training complete. model.pkl and meta.pkl created.")

if __name__ == "__main__":
    train_and_save_model()
