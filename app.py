import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="centered")
st.markdown("<h1 style='text-align:center;color:#b30000;'>❤️ Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.divider()

def map_input(value, mapping): return mapping[value]

age = st.number_input("Age", 18, 100)
sex = map_input(st.selectbox("Gender", ["Male","Female"]), {"Male":1,"Female":0})
cp = map_input(st.selectbox("Chest Pain", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"]),
               {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3})
trestbps = st.number_input("Resting BP",80,200)
chol = st.number_input("Cholesterol",100,600)
fbs = map_input(st.selectbox("Fasting Blood Sugar >120 mg/dl", ["No","Yes"]), {"No":0,"Yes":1})
restecg = map_input(st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"]),
                    {"Normal":0,"ST-T Abnormality":1,"Left Ventricular Hypertrophy":2})
thalach = st.number_input("Max Heart Rate",60,220)
exang = map_input(st.selectbox("Exercise Angina", ["No","Yes"]), {"No":0,"Yes":1})
oldpeak = st.number_input("ST Depression (Oldpeak)",0.0,6.0)
slope = map_input(st.selectbox("Slope of ST Segment", ["Upsloping","Flat","Downsloping"]),
                  {"Upsloping":0,"Flat":1,"Downsloping":2})
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = map_input(st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversible Defect"]),
                 {"Normal":0,"Fixed Defect":1,"Reversible Defect":2})

st.divider()

threshold = st.slider("Prediction Threshold (Lower = more sensitive)",0.30,0.70,0.50)

if st.button("🔍 Predict Heart Disease Risk"):
    input_df = pd.DataFrame([{
        "age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,
        "restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,
        "slope":slope,"ca":ca,"thal":thal
    }])[features]

    prob = model.predict_proba(scaler.transform(input_df))[0][1]
    risk_percent = prob*100

    # Risk label
    if risk_percent<40: risk_label,color,advice="LOW RISK","green","Maintain healthy lifestyle."
    elif risk_percent<70: risk_label,color,advice="MEDIUM RISK","orange","Consult a doctor & improve lifestyle."
    else: risk_label,color,advice="HIGH RISK","red","Immediate medical consultation advised."

    st.markdown(f"<h2 style='color:{color};'>🩺 Risk Level: {risk_label}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>📊 Heart Disease Probability: {risk_percent:.2f}%</h3>", unsafe_allow_html=True)
    st.info(f"👨‍⚕️ Advice: {advice}")

    # Threshold decision
    if prob>=threshold: st.error("⚠️ Heart Disease Detected")
    else: st.success("✅ No Heart Disease Detected")

    # Feature importance
    st.divider()
    st.subheader("📌 Feature Importance")
    fi_df = pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance")
    fig,ax=plt.subplots(figsize=(7,5)); ax.barh(fi_df["Feature"],fi_df["Importance"],color="#b30000"); ax.set_xlabel("Importance"); st.pyplot(fig)
