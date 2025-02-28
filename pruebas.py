import cv2 as cv
import mediapipe as mp
import joblib
import numpy as np

# Cargar el modelo previamente entrenado
model = joblib.load('modelo_gestos.pkl')

# Inicializar Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Abrir la cámara
cap = cv.VideoCapture(0)

# Función para predecir el gesto
def predict_gesture(landmarks):
    # Verifica si hay la cantidad correcta de valores en los landmarks
    if len(landmarks) == model.n_features_in_:  # Asegura la coincidencia de dimensiones
        landmarks = np.array(landmarks).reshape(1, -1)  # Convertir a formato adecuado
        prediction = model.predict(landmarks)
        return prediction[0]  # Retorna la etiqueta del gesto
    else:
        return "Error: Coordenadas insuficientes"

print("Reconocimiento de gestos en vivo. Presiona 'q' para salir.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Detectar y dibujar las manos en la imagen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convertir puntos de referencia a una lista
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predecir el gesto solo si se detectaron landmarks suficientes
            predicted_gesture = predict_gesture(landmarks)
            cv.putText(img, f"Gesto: {predicted_gesture}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen
    cv.imshow('Reconocimiento de Gestos', img)

    # Salir con la tecla 'q'
    if cv.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv.destroyAllWindows()
