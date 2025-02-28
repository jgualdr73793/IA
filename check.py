import cv2 as cv
import mediapipe as mp
import csv
import time

# Inicializar mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Abrir la cámara
cap = cv.VideoCapture(0)

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

# Crear y abrir el archivo CSV en modo de escritura
with open('hand_gesture_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Encabezado del CSV
    header = [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    writer.writerow(header)

    print("Comienza la recolección de datos. Presiona 'q' para salir.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: No se puede capturar el frame.")
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

                # Mostrar imagen
                cv.imshow('Image', img)

                # Pedir la etiqueta del gesto
                label = input("Introduce la etiqueta para el gesto actual (ej: puño, abierto, victoria): ")

                # Confirmar si se quiere comenzar a capturar el gesto
                confirm = input(f"¿Confirmas la captura para el gesto '{label}'? (s para sí, n para no): ")
                if confirm.lower() == 's':
                    print("Capturando datos. Mueve la mano para diferentes posiciones...")
                    start_time = time.time()
                    capture_interval = 0.5  # Intervalo de captura en segundos
                    frame_count = 0  # Contador de frames guardados

                    while frame_count < 100:  # Limitar a 100 frames por gesto
                        success, img = cap.read()
                        if not success:
                            print("Error: No se puede capturar el frame.")
                            break

                        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                        results = hands.process(img_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                                # Convertir puntos de referencia a una lista
                                landmarks = []
                                for lm in hand_landmarks.landmark:
                                    landmarks.extend([lm.x, lm.y, lm.z])

                                # Guardar los datos cada 0.5 segundos
                                if time.time() - start_time >= capture_interval:
                                    landmarks.append(label)
                                    writer.writerow(landmarks)
                                    frame_count += 1  # Incrementar contador de frames
                                    print(f"Gesto '{label}' guardado en el CSV. Total frames: {frame_count}/100")
                                    start_time = time.time()

                        cv.imshow('Image', img)

                        # Salir con la tecla 'q'
                        if cv.waitKey(1) == ord('q'):
                            break

                else:
                    print("Captura cancelada. Ingresa un nuevo gesto.")

        # Salir con la tecla 'q'
        if cv.waitKey(1) == ord('q'):
            break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv.destroyAllWindows()
