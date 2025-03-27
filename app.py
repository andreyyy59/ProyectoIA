import streamlit as st
import pandas as pd
import joblib
import numpy as np

scaler = joblib.load('scaler.pkl')
modelo = joblib.load('gaussian_nb_model.pkl')

st.set_page_config(page_title="Predictor de Felicidad", page_icon="😊")

st.title("🧠 Modelo calcular puntaje de felicidad con IA 🤖")
st.subheader("Realizado por Andrey Morales y Johan Latorre 👥")

st.markdown("""
    ### 🌟 Bienvenido a nuestra Aplicación de Predicción de Felicidad

    Esta herramienta innovadora utiliza inteligencia artificial para estimar tu nivel de felicidad 
    basándose en factores clave de tu estilo de vida. Al ingresar datos como tus horas de sueño, 
    horas de trabajo, tiempo en pantalla e interacción social, nuestro modelo puede predecir 
    tu potencial nivel de bienestar emocional.

    #### ¿Cómo funciona? 🤔
    - Desliza los controles para ingresar tus datos
    - Presiona "Predecir Felicidad"
    - Obtén un puntaje personalizado de 0 a 10
""")

st.sidebar.header("📊 Ingresa tus datos")
age = st.sidebar.slider("Edad", 18, 80, 30)
sleep_hours = st.sidebar.slider("Horas de Sueño", 0, 10, 7)
work_hours = st.sidebar.slider("Horas de Trabajo por Semana", 1, 80, 40)
screen_time = st.sidebar.slider("Tiempo en Pantalla (Horas)", 1, 10, 4)
social_interaction = st.sidebar.slider("Puntaje de Interacción Social", 1, 10, 7)

if st.sidebar.button("🔮 Predecir Felicidad"):
    data = pd.DataFrame({
        'Sleep Hours': [sleep_hours],
        'Work Hours per Week': [work_hours],
        'Screen Time per Day (Hours)': [screen_time],
        'Social Interaction Score': [social_interaction],
        'Age': [age]
    })

    if "Happiness Score" in scaler.feature_names_in_:
        data["Happiness Score"] = 0  

    data = data[scaler.feature_names_in_]

    data_normalizada = scaler.transform(data)

    if "Happiness Score" in data:
        data_normalizada = data_normalizada[:, :-1]

    prediccion = modelo.predict(data_normalizada)[0]

    if prediccion <= 5:
        color = '#3498db'  
        emoji = '😔'
        mensaje = "¡Necesitas mejorar algunos aspectos de tu vida!"
    else:
        color = '#e74c3c'  
        emoji = '😄'
        mensaje = "¡Excelente nivel de felicidad!"

    st.markdown(f"""
    ### Resultado de tu Predicción de Felicidad {emoji}

    <div style="
        background-color: {color}; 
        color: white; 
        padding: 20px; 
        border-radius: 10px;
        text-align: center;
        font-size: 20px;">
        <b>Puntaje de Felicidad:</b> {prediccion:.2f}/10  
        <br>
        {mensaje}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("© UNAB 2025")
