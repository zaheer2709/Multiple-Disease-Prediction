import streamlit as st
import pickle
import urllib.request
import os
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# GitHub raw URLs (replace with your GitHub repo URL)
GITHUB_REPO_URL = "https://github.com/mohammedarifsn12/multiple_disease_prediction.git"
MODEL_FILES = {
    "diabetes": "diabetes.sav",
    "heart": "heart.sav",
    "parkinson": "parkinsons.sav"
}

# Download function
def download_model(model_name):
    model_path = f"./{MODEL_FILES[model_name]}"
    if not os.path.exists(model_path):
        url = GITHUB_REPO_URL + MODEL_FILES[model_name]
        urllib.request.urlretrieve(url, model_path)
    return model_path

# Load models
diabetes_model = pickle.load(open(download_model("diabetes"), 'rb'))
heart_disease_model = pickle.load(open(download_model("heart"), 'rb'))
parkinsons_model = pickle.load(open(download_model("parkinson"), 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=0)

    with col2:
        Glucose = st.number_input('Glucose Level', value=0)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value', value=0)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value', value=0)

    with col2:
        Insulin = st.number_input('Insulin Level', value=0)

    with col3:
        BMI = st.number_input('BMI value', value=0.0)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=0.0)

    with col2:
        Age = st.number_input('Age of the Person', value=0)

    if st.button('Diabetes Test Result'):
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = diabetes_model.predict(user_input)
        st.success("The person is diabetic" if prediction[0] == 1 else "The person is not diabetic")

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', value=0)

    with col2:
        sex = st.selectbox('Sex', [0, 1])  # 0: Female, 1: Male

    with col3:
        cp = st.number_input('Chest Pain types', value=0)

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', value=0)

    with col2:
        chol = st.number_input('Serum Cholesterol in mg/dl', value=0)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])

    with col1:
        restecg = st.number_input('Resting Electrocardiographic results', value=0)

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', value=0)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', [0, 1])

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', value=0.0)

    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment', value=0)

    with col3:
        ca = st.number_input('Major vessels colored by fluoroscopy', value=0)

    with col1:
        thal = st.number_input('Thal (0=normal, 1=fixed defect, 2=reversible defect)', value=0)

    if st.button('Heart Disease Test Result'):
        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = heart_disease_model.predict(user_input)
        st.success("The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease")

# Parkinson's Prediction
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    input_values = []
    cols = st.columns(5)

    for i, feature in enumerate(features):
        with cols[i % 5]:
            input_values.append(st.number_input(feature, value=0.0))

    if st.button("Parkinson's Test Result"):
        user_input = [input_values]
        prediction = parkinsons_model.predict(user_input)
        st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")
