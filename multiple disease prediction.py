import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

diabetes_model = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/diabetes_model.sav", 'rb'))
diabetes_scaler = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/diabetes_scaler.sav", 'rb'))

parkinsons_model = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/parkinsons_model.sav", 'rb'))
parkinsons_scaler = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/parkinsons_scaler.sav", 'rb'))

decision_tree_model = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/decision_tree_model.pkl", 'rb'))
random_forest_model = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/random_forest_model.pkl", 'rb'))
naive_bayes_model = pickle.load(open("C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/naive_bayes_model.pkl", 'rb'))

stacking_model = pickle.load(open('C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/stacking_model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/scaler.pkl', 'rb'))
label_encoders = pickle.load(open('C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/label_encoders.pkl', 'rb'))

autism_model = pickle.load(open('C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/autism_model.pkl', 'rb'))
autism_encoders = pickle.load(open('C:/Users/shubh/OneDrive/Desktop/Multiple Disease Prediction/saved models/autism_encoders.pkl', 'rb'))




l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
      'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
      'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
      'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
      'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
      'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
      'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
      'bladder_discomfort', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
      'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

disease_map = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis', 4: 'Drug Reaction',
    5: 'Peptic ulcer diseae', 6: 'AIDS', 7: 'Diabetes', 8: 'Gastroenteritis', 9: 'Bronchial Asthma',
    10: 'Hypertension', 11: 'Migraine', 12: 'Cervical spondylosis', 13: 'Paralysis (brain hemorrhage)',
    14: 'Jaundice', 15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid', 19: 'hepatitis A',
    20: 'Hepatitis B', 21: 'Hepatitis C', 22: 'Hepatitis D', 23: 'Hepatitis E', 24: 'Alcoholic hepatitis',
    25: 'Tuberculosis', 26: 'Common Cold', 27: 'Pneumonia', 28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack',
    30: 'Varicose veins', 31: 'Hypothyroidism', 32: 'Hyperthyroidism', 33: 'Hypoglycemia', 34: 'Osteoarthristis',
    35: 'Arthritis', 36: '(vertigo) Paroymsal  Positional Vertigo', 37: 'Acne', 38: 'Urinary tract infection',
    39: 'Psoriasis', 40: 'Impetigo'
}



with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           
                           ["Diabetes Prediction",
                            "Parkisons Prediction",
                            "Disease Prediction using ML Models",
                            "Heart Disease Prediction"],
                           
                           icons = ['activity', 'person', 'hospital', 'heart-pulse-fill'],
                           
                            default_index = 0)
    
if (selected == 'Diabetes Prediction'):
    st.title("Diabetes Prediction using ML")
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        
    with col2:
        Glucose = st.text_input("Glucose Level")
        
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
        
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
        
    with col2:
        Insulin = st.text_input("Insulin Level")
        
    with col3:
        BMI = st.text_input("BMI Value")
        
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
        
    with col2:
        Age = st.text_input("Age of the Person")
        
    
    
    diab_diagnosis = ""
    
    if st.button("Diabetes Test Result"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=float)
        
        scaled_input_data = diabetes_scaler.transform(input_data)
        
        diab_prediction = diabetes_model.predict(scaled_input_data)
    
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The Person is diabetic'
        else:
            diab_diagnosis = 'The Person is not diabetic'
            
    st.success(diab_diagnosis)
    
    
    
    
if (selected == 'Parkisons Prediction'):
    st.title("Parkisons Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        MDVP_Fo_Hz = st.text_input("MDVP:Fo(Hz)")
        
    with col2:
        MDVP_Fhi_Hz = st.text_input("MDVP:Fhi(Hz)")
        
    with col3:
        MDVP_Flo_Hz = st.text_input("MDVP:Flo(Hz)")
        
    with col4:
        MDVP_Jitter_percent = st.text_input("MDVP:Jitter(%)")
        
    with col5:
        MDVP_Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        
    with col1:
        MDVP_RAP = st.text_input("MDVP:RAP")
        
    with col2:
        MDVP_PPQ = st.text_input("MDVP:PPQ")
        
    with col3:
        Jitter_DDP = st.text_input("Jitter:DDP")
        
    with col4:
        MDVP_Shimmer = st.text_input("MDVP:Shimmer")
        
    with col5:
        MDVP_Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
        
    with col1:
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
        
    with col2:
        Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
        
    with col3:
        MDVP_APQ = st.text_input("MDVP:APQ")
        
    with col4:
        Shimmer_DDA = st.text_input("Shimmer:DDA")
        
    with col5:
        NHR = st.text_input("NHR")
        
    with col1:
        HNR = st.text_input("HNR")
        
    with col2:
        RPDE = st.text_input("RPDE")
    
    with col3:    
        DFA = st.text_input("DFA")
        
    with col4:
        spread1 = st.text_input("spread1")
        
    with col5:
        spread2 = st.text_input("spread2")
        
    with col1:
        D2 = st.text_input("D2")
        
    with col2:
        PPE = st.text_input("PPE")
        
    
    
    park_diagnosis = ""
    
    if st.button("Parkinson's Test Result"):
        input_data = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]], dtype=float)
        
        scaled_input_data = parkinsons_scaler.transform(input_data)
        
        park_prediction = parkinsons_model.predict(scaled_input_data)
    
        if (park_prediction[0] == 1):
            park_diagnosis = 'The Person is having Parkinsons'
        else:
            park_diagnosis = 'The Person is not having Parkinsons'
            
    st.success(park_diagnosis)


if selected == "Disease Prediction using ML Models":
    st.title("Disease Prediction by Symptoms")
    
    symptoms_input = []
    
    for i in range(5):  # Loop to show 5 symptoms dropdowns
        symptom = st.selectbox(f"Choose symptom {i+1}", l1)
        symptoms_input.append(symptom)
    
    symptoms_input_binary = [1 if symptom in symptoms_input else 0 for symptom in l1]
    
    symptoms_input_array = np.array([symptoms_input_binary]).astype(int)
    
    if st.button("Prediction 1"):
        prediction_dt = decision_tree_model.predict(symptoms_input_array)
        disease_name = disease_map.get(prediction_dt[0], "Unknown disease")
        st.success(f"Predicted Disease: {disease_name}")
    
    if st.button("Prediction 2"):
        prediction_rf = random_forest_model.predict(symptoms_input_array)
        disease_name = disease_map.get(prediction_rf[0], "Unknown disease")
        st.success(f"Predicted Disease: {disease_name}")
    
    if st.button("Prediction 3"):
        prediction_nb = naive_bayes_model.predict(symptoms_input_array)
        disease_name = disease_map.get(prediction_nb[0], "Unknown disease")
        st.success(f"Predicted Disease: {disease_name}")



if selected == "Heart Disease Prediction":

    st.title("Heart Disease Prediction using Stacking Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input("Age")
    with col2:
        Sex = st.selectbox("Sex", ['M', 'F'])
    with col3:
        ChestPainType = st.selectbox("ChestPainType", ['TA', 'ATA', 'NAP', 'ASY'])
    
    with col1:
        RestingBP = st.text_input("Resting Blood Pressure")
    with col2:
        Cholesterol = st.text_input("Cholesterol")
    with col3:
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    
    with col1:
        RestingECG = st.selectbox("RestingECG", ['Normal', 'ST', 'LVH'])
    with col2:
        MaxHR = st.text_input("Max Heart Rate")
    with col3:
        ExerciseAngina = st.selectbox("ExerciseAngina", ['Y', 'N'])
    
    with col1:
        Oldpeak = st.text_input("Oldpeak (Depression)")
    with col2:
        ST_Slope = st.selectbox("ST_Slope", ['Up', 'Flat', 'Down'])


    if st.button("Heart Disease Test Result"):

        input_data = {
            'Age': [Age],
            'Sex': [Sex],
            'ChestPainType': [ChestPainType],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBS],
            'RestingECG': [RestingECG],
            'MaxHR': [MaxHR],
            'ExerciseAngina': [ExerciseAngina],
            'Oldpeak': [Oldpeak],
            'ST_Slope': [ST_Slope]
        }

        input_df = pd.DataFrame(input_data)

        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

        numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns].astype(float))

        expected_order = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        input_df = input_df[expected_order]  

        prediction = stacking_model.predict(input_df)

        if prediction[0] == 1:
            st.success("The Person is likely to have heart disease")
        else:
            st.success("The Person is unlikely to have heart disease")
































