import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ======== FUNCIÓN DE PREDICCIÓN =========
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# ======== CONFIGURACIÓN DE LA APP ========
st.set_page_config(page_title='Reconocimiento de Dígitos', layout='wide')

# 🎨 Estilos personalizados (COLORES VISIBLES)
st.markdown("""
<style>
/* Fondo principal */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8E8FF 0%, #E3F2FD 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F3E5F5;
    color: #4A148C;
    font-weight: bold;
}

/* Contenedor principal */
.block-container {
    background-color: #FFFFFFEE;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 20px rgba(0,0,0,0.15);
}

/* Títulos */
h1 {
    color: #6A1B9A;
    text-align: center;
    font-family: 'Trebuchet MS', sans-serif;
}

h2, h3, p, label, span {
    color: #333333;
    font-family: 'Verdana';
}

/* Botón principal */
div.stButton > button {
    background: linear-gradient(90deg, #AB47BC, #8E24AA);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #8E24AA, #6A1B9A);
    transform: scale(1.05);
}

/* Resultado */
.success {
    background-color: #C8E6C9 !important;
    color: #1B5E20 !important;
    border-radius: 10px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ======== ENCABEZADO ========
st.markdown("""
<h1>🔢 Reconocimiento de Dígitos escritos a mano</h1>
<p style='text-align:center; font-size:18px; color:#555;'>
Dibuja un número y deja que la red neuronal adivine cuál es.
</p>
""", unsafe_allow_html=True)

# ======== PANEL DE DIBUJO ========
st.subheader("✏️ Dibuja el dígito en el panel y presiona 'Predecir'")

drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# ======== BOTÓN DE PREDICCIÓN ========
predict_button = st.button('🔮 ¡Predecir ahora!')
if predict_button:
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.markdown(f"<div class='success'><h2>🎯 El dígito reconocido es: {res}</h2></div>", unsafe_allow_html=True)
    else:
        st.warning('Por favor dibuja en el canvas el dígito antes de predecir.')

# ======== SIDEBAR ========
st.sidebar.title("💡 Acerca de")
st.sidebar.text("Esta aplicación usa una Red Neuronal")
st.sidebar.text("para reconocer dígitos escritos a mano.")
st.sidebar.text("Modelo basado en el trabajo de Vinay Uniyal.")


