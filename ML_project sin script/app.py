import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Bitcoin",
    page_icon="📈",
    layout="wide"
)

# Función para centrar texto con HTML/CSS
def centered_header(text, level=1):
    if level == 1:
        return st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)
    elif level == 2:
        return st.markdown(f"<h2 style='text-align: center;'>{text}</h2>", unsafe_allow_html=True)
    elif level == 3:
        return st.markdown(f"<h3 style='text-align: center;'>{text}</h3>", unsafe_allow_html=True)
    else:
        return st.markdown(f"<p style='text-align: center; font-size: 20px;'>{text}</p>", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def load_data():
    """Cargar el dataframe procesado"""
    return pd.read_csv('data/df_selected.csv')

@st.cache_resource
def load_model():
    """Cargar el modelo entrenado"""
    with open('model/randomforest.pkl', 'rb') as file:
        return pickle.load(file)

def get_current_bitcoin_price():
    """Obtener el precio actual de Bitcoin desde Yahoo Finance"""
    btc_data = yf.Ticker("BTC-USD").history(period="1d")
    return btc_data['Close'].iloc[-1]

def get_bitcoin_historical_data():
    """Obtener datos históricos de Bitcoin para graficar"""
    return yf.Ticker("BTC-USD").history(period="6mo")

def prepare_features_for_prediction(price):
    """Preparar features para la predicción con el modelo"""
    features = {
        'close_btc': price,
        'close_qqq': yf.Ticker("QQQ").history(period="1d")['Close'].iloc[-1],
        'difficulty': 0,  # Necesitarías una API para esto
        'close_VIX': yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1],
        'volume_qqq': yf.Ticker("QQQ").history(period="1d")['Volume'].iloc[-1],
        'btc_change': 0  # Se calcularía con datos históricos
    }
    return pd.DataFrame([features])

# Cargar datos y modelo
df = load_data()
model = load_model()

# Obtener la página actual de los parámetros de URL
page = st.query_params.get("page", "home")

# Función para navegar a una página
def navigate_to(page_name):
    st.query_params["page"] = page_name

# Botón para volver al inicio (añadido a todas las páginas excepto 'home')
def home_button():
    if st.button("🏠 Volver a Inicio"):
        navigate_to("home")
        st.rerun()

# PÁGINA DE INICIO
if page == "home":
    # Título y descripción
    centered_header("📊 Sistema de Predicción de Bitcoin")
    st.markdown("""
    <p style='text-align:center'>
    Esta aplicación utiliza machine learning para predecir el movimiento del precio de Bitcoin para el día siguiente.
    </p>
    """, unsafe_allow_html=True)
    
    # SECCIÓN PRINCIPAL - GRÁFICO DE PRECIO
    centered_header("Precio actual de Bitcoin", level=2)

    # Obtener datos históricos para graficar
    hist_data = get_bitcoin_historical_data()
    current_price = hist_data['Close'].iloc[-1]

    # Gráfico de precio
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_data.index, hist_data['Close'])
    ax.set_title('Precio de Bitcoin (Últimos 6 meses)')
    ax.set_ylabel('Precio (USD)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Precio actual como métrica centralizada con estilo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style='text-align: center; padding: 10px;'>
                <h3>BTC-USD</h3>
                <h2 style='color: #0068c9;'>${current_price:,.2f}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # BOTONES DE NAVEGACIÓN
    centered_header("Explorar más información", level=3)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚦 Predicción Semáforo", use_container_width=True):
            navigate_to("prediction")
            st.rerun()
    with col2:
        if st.button("📊 Datos Históricos", use_container_width=True):
            navigate_to("historical")
            st.rerun()
    with col3:
        if st.button("🧠 Modelo de Predicción", use_container_width=True):
            navigate_to("model")
            st.rerun()

# PÁGINA DE PREDICCIÓN
elif page == "prediction":
    centered_header("🚦 Predicción Semáforo para el Día Siguiente")
    home_button()
    
    try:
        # Obtener precio actual
        current_price = get_current_bitcoin_price()
        
        # Preparar datos para predicción
        X_pred = prepare_features_for_prediction(current_price)
        
        # Hacer predicción
        prediction = model.predict(X_pred)[0]
        
        # Mostrar resultado con color apropiado
        col1, col2 = st.columns([1, 3])
        with col1:
            if prediction == 'verde':
                st.markdown('<div style="background-color:#28a745;color:white;padding:30px;border-radius:50%;text-align:center;font-size:24px;">🟢</div>', unsafe_allow_html=True)
            elif prediction == 'amarillo':
                st.markdown('<div style="background-color:#ffc107;color:white;padding:30px;border-radius:50%;text-align:center;font-size:24px;">🟡</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color:#dc3545;color:white;padding:30px;border-radius:50%;text-align:center;font-size:24px;">🔴</div>', unsafe_allow_html=True)
        
        with col2:
            if prediction == 'verde':
                st.success("### VERDE - Tendencia alcista esperada")
                st.write("Se espera que el precio de Bitcoin suba en el próximo día.")
            elif prediction == 'amarillo':
                st.warning("### AMARILLO - Mercado lateral esperado")
                st.write("Se espera poca variación en el precio de Bitcoin para el próximo día.")
            else:
                st.error("### ROJO - Tendencia bajista esperada")
                st.write("Se espera que el precio de Bitcoin baje en el próximo día.")
    
        # Comparativa de rendimientos
        centered_header("Comparativa de rendimientos", level=2)
        st.markdown("<p style='text-align:center'>Rendimiento acumulado de las estrategias (RandomForest vs Buy & Hold)</p>", unsafe_allow_html=True)
        
        try:
            img = plt.imread('images/backtest_comparison.png')
            st.image(img, caption="", use_container_width=True)
        except:
            st.write("Imagen de backtest no encontrada. Asegúrate de generar y guardar el gráfico de backtest en 'images/backtest_comparison.png'")
            
    except Exception as e:
        st.error(f"Error obteniendo datos actuales o realizando predicción: {e}")

# PÁGINA DE DATOS HISTÓRICOS
elif page == "historical":
    centered_header("📊 Datos Históricos Utilizados")
    home_button()
    
    st.dataframe(df)
    
    # Mostrar estadísticas descriptivas
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.write(df.describe())
    
    # Permitir visualizar correlaciones
    if st.checkbox("Mostrar matriz de correlación"):
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# PÁGINA DEL MODELO
elif page == "model":
    centered_header("🧠 Modelo Random Forest")
    home_button()
    
    st.markdown("<p style='text-align:center'>Este modelo fue seleccionado por tener la mejor capacidad de generalización entre los modelos evaluados.</p>", unsafe_allow_html=True)
    
    # Mostrar matriz de confusión
    centered_header("Matriz de confusión", level=2)
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img = plt.imread('images/confusion_matrix_randomforest.png')
            st.image(img, caption="", use_container_width=True)
    except:
        st.write("Imagen de matriz de confusión no encontrada. Asegúrate de guardarla en 'images/confusion_matrix_randomforest.png'")
    
    # Mostrar importancia de features
    centered_header("Importancia de las características", level=2)
    try:
        # Extraer importancia de features si el modelo lo permite
        current_price = get_current_bitcoin_price()
        X_pred = prepare_features_for_prediction(current_price)
        
        if hasattr(model[-1], 'feature_importances_'):
            importances = model[-1].feature_importances_
            feature_names = X_pred.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            plt.title('Importancia de las características')
            st.pyplot(fig)
            
            # Texto explicativo sobre importancia de características
            st.markdown("""
            <div style='background-color:#f8f9fa; padding:15px; border-radius:10px; margin-top:20px;'>
                <h4 style='text-align:center'>¿Qué significa la importancia de características?</h4>
                <p>En un modelo Random Forest, <strong>la importancia de las características</strong> indica cuánto contribuye cada variable a la precisión de las predicciones:</p>
                <ul>
                    <li><strong>Mayor valor</strong>: La característica tiene un impacto más fuerte en la predicción del comportamiento del Bitcoin.</li>
                    <li><strong>Menor valor</strong>: La característica tiene menos influencia en la decisión del modelo.</li>
                </ul>
                <p>Esto nos ayuda a entender qué factores del mercado son más relevantes para predecir si el precio del Bitcoin subirá, bajará o se mantendrá estable.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("La importancia de features no está disponible para este modelo")
    except Exception as e:
        st.write(f"Error mostrando importancia de features: {e}")