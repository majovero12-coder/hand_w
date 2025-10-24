import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
# 🌈 Fondo de color y estilo general
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F0F8FF;
}
[data-testid="stSidebar"] {
    background-color: #E3F2FD;
}
.block-container {
    padding: 2rem 3rem;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
    background-color: #FFFFFF;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# 🎨 Encabezado visual
st.markdown("""
    <h1 style='text-align: center; color: #4B9CD3;'>
        ✨ Reconocimiento de Dígitos ✋
    </h1>
    <p style='text-align: center; color: #6C757D; font-size:18px;'>
        Dibuja un número y deja que la red neuronal adivine qué escribiste.
    </p>
""", unsafe_allow_html=True)

st.subheader("Dibuja el digito en el panel  y presiona  'Predecir'")

# Add canvas component
# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF' # Set background color to white
bg_color = '#000000'

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Add "Predict Now" button
predict_button = st.button('🔮 ¡Predecir ahora!')
if predict_button:
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('El Digito es : ' + str(res))
    else:
        st.success(f"🎯 El dígito reconocido es: **{res}**")

# Add sidebar
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalua ")
st.sidebar.text("la capacidad de un RNA de reconocer") 
st.sidebar.text("digitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")
#st.sidebar.text("GitHub Repository")
#st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")

