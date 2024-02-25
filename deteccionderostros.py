import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Cargo el clasificador preentrenado
cascada = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

def detect_faces(image):
    """Detecto rostros en una imagen."""
    image_np = np.array(image.convert('RGB'))  # Convierto PIL Image a NumPy array
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # Convierto a escala de grises
    faces = cascada.detectMultiScale(gray, 1.1, 5)  # Detecto rostros
    conteo = 0  # Inicio el contador
   

    for (x, y, w, h) in faces:
        conteo += 1  #Incremento el contador para cada rostro
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dibujo rectángulos alrededor de los rostros
        image_np = cv2.putText(image_np, "Rostro #" + str(conteo), (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    return image_np, conteo

def main():
    """Función principal de la aplicación Streamlit."""
    st.title("Detector de Rostros")

    mode_selection = st.radio("Elige el modo de entrada:", ["Subir Imagen", "Usar Cámara"])

    if mode_selection == "Subir Imagen":
        uploaded_image = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Imagen subida', use_column_width=True)
            if st.button('Detectar'):
                result_image, conteo = detect_faces(image)
                st.write(f"Rostros detectados: {conteo}")
                st.image(result_image, caption='Imagen con rostros detectados', use_column_width=True)
    else:
        camera_image = st.camera_input("Toma una foto")
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption='Imagen capturada', use_column_width=True)
            if st.button('Detectar'):
                result_image, conteo = detect_faces(image)
                st.write(f"Rostros detectados: {conteo}")
                st.image(result_image, caption='Imagen con rostros detectados', use_column_width=True)

if __name__ == "__main__":
    main()

