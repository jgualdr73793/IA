# creación entorno virtual -> python -m venv tutorial-env
# Activación Entorno Virtual -> tutorial-env\Scripts\activate

import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Cargar el escalador y el modelo
scaler = joblib.load('scaler.pkl')
model = joblib.load('modelo_entrenado.pkl')

# Función para procesar la información y hacer la predicción
def realizar_prediccion():
    try:
        # Capturamos los valores ingresados (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        pregnancies = float(entry_pregnancies.get())
        glucose = float(entry_glucose.get())
        blood_pressure = float(entry_blood_pressure.get())
        skin_thickness = float(entry_skin_thickness.get())
        insulin = float(entry_insulin.get())
        bmi = float(entry_bmi.get())
        diabetes_pedigree_function = float(entry_diabetes_pedigree_function.get())
        age = float(entry_age.get())

        # Creamos el array con los datos ingresados (ajustar dimensiones según modelo)
        xn = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
        Xn_std = scaler.transform(xn.reshape(1, -1))  # Escalado y redimensionado

        # Realizamos la predicción
        resul = model.predict(Xn_std)

        # Mostramos el resultado en un mensaje emergente
        messagebox.showinfo("Resultado", f"Los datos ingresados corresponden a una predicción de {resul[0]}")
    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Predicción de Diabetes")

# Etiquetas y entradas de texto
tk.Label(root, text="Embarazos:").grid(row=0, column=0, padx=10, pady=10)
entry_pregnancies = tk.Entry(root)
entry_pregnancies.grid(row=0, column=1)

tk.Label(root, text="Glucosa:").grid(row=1, column=0, padx=10, pady=10)
entry_glucose = tk.Entry(root)
entry_glucose.grid(row=1, column=1)

tk.Label(root, text="Presión Sanguínea:").grid(row=2, column=0, padx=10, pady=10)
entry_blood_pressure = tk.Entry(root)
entry_blood_pressure.grid(row=2, column=1)

tk.Label(root, text="Grosor de la Piel:").grid(row=3, column=0, padx=10, pady=10)
entry_skin_thickness = tk.Entry(root)
entry_skin_thickness.grid(row=3, column=1)

tk.Label(root, text="Insulina:").grid(row=4, column=0, padx=10, pady=10)
entry_insulin = tk.Entry(root)
entry_insulin.grid(row=4, column=1)

tk.Label(root, text="IMC:").grid(row=5, column=0, padx=10, pady=10)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=5, column=1)

tk.Label(root, text="Función de Pedigrí de Diabetes:").grid(row=6, column=0, padx=10, pady=10)
entry_diabetes_pedigree_function = tk.Entry(root)
entry_diabetes_pedigree_function.grid(row=6, column=1)

tk.Label(root, text="Edad:").grid(row=7, column=0, padx=10, pady=10)
entry_age = tk.Entry(root)
entry_age.grid(row=7, column=1)

# Botón para ejecutar la predicción
btn_predecir = tk.Button(root, text="Predecir", command=realizar_prediccion)
btn_predecir.grid(row=8, column=0, columnspan=2, pady=10)

# Iniciar la ventana
root.mainloop()