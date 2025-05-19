from flask import Flask, render_template, request, redirect, flash, url_for
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)
app.secret_key = 'clave_secreta_segura'  


# Cargar modelos
modelos = {
    'Regresión Logística': pickle.load(open('models/logistic_model.pkl', 'rb')),
    'Red Neuronal': pickle.load(open('models/ann_model.pkl', 'rb')),
    'SVM': pickle.load(open('models/svm_model.pkl', 'rb')),
    'Mapa Cognitivo Difuso': pickle.load(open('models/fcm_model.pkl', 'rb'))
}


columnas = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
            'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
            'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25',
            'C26', 'C27', 'C28', 'C29', 'C30']


campos_legibles = {
    'C1': 'Edad',
    'C2': 'IMC',
    'C3': 'Edad gestacional al parto',
    'C4': 'Gravidez',
    'C5': 'Paridad',
    'C6': 'Síntomas iniciales (0=edema, 1=hipertensión, 2=FGR)',
    'C7': 'Edad gestacional del inicio de síntomas',
    'C8': 'Días desde síntomas hasta parto',
    'C9': 'Edad gestacional del inicio de hipertensión',
    'C10': 'Días desde hipertensión hasta parto',
    'C11': 'Edad gestacional del inicio de edema',
    'C12': 'Días desde edema hasta parto',
    'C13': 'Edad gestacional del inicio de proteinuria',
    'C14': 'Días desde proteinuria hasta parto',
    'C15': 'Tratamiento expectante (0=No, 1=Sí)',
    'C16': 'Antihipertensivos antes de hospitalización (0=No, 1=Sí)',
    'C17': 'Antecedentes (0=No, 1=Hipertensión, 2=PCOS)',
    'C18': 'Presión sistólica máxima',
    'C19': 'Presión diastólica máxima',
    'C20': 'Razón del parto (0=HELLP, 1=Distress fetal, etc.)',
    'C21': 'Modo de parto (0=Cesárea, 1=Parto vaginal)',
    'C22': 'BNP máximo',
    'C23': 'Creatinina máxima',
    'C24': 'Ácido úrico máximo',
    'C25': 'Proteinuria máxima',
    'C26': 'Proteína total máxima',
    'C27': 'Albúmina máxima',
    'C28': 'ALT máxima',
    'C29': 'AST máxima',
    'C30': 'Plaquetas máximas'
}

def predict_with_fcm(modelo_fcm, X_input):
    scaler = modelo_fcm['scaler']
    centers = modelo_fcm['centers']
    X_scaled = scaler.transform(X_input)
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
    y_pred = np.argmin(distances, axis=1)
    return y_pred

@app.route('/')
def home():
    return render_template('index.html', columnas=columnas, campos_legibles=campos_legibles)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    modelo_nombre = request.form['modelo']
    entrada = [float(request.form[col]) for col in columnas]
    entrada_np = np.array(entrada).reshape(1, -1)

    if modelo_nombre == 'Mapa Cognitivo Difuso':
        modelo = modelos[modelo_nombre]
        y_pred = predict_with_fcm(modelo, entrada_np)
    else:
        modelo = modelos[modelo_nombre]
        y_pred = modelo.predict(entrada_np)

    resultado = "FGR" if y_pred[0] == 1 else "Normal"

    return render_template('resultado.html',
                       resultado_manual=resultado,
                       modelo_seleccionado=modelo_nombre)


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    modelo_nombre = request.form['modelo']
    archivo = request.files['archivo']

    if not archivo.filename.endswith('.xlsx'):
        flash("❌ Archivo no compatible. Por favor sube un archivo .xlsx", "danger")
        return redirect(url_for('home'))

    df = pd.read_excel(archivo)

    if not all(col in df.columns for col in columnas + ['C31']):
        flash("⚠️ El archivo debe contener todas las columnas C1 a C30 y C31 (etiqueta verdadera).", "warning")
        return redirect(url_for('home'))

    X = df[columnas]
    y_true = df['C31']

    if modelo_nombre == 'Mapa Cognitivo Difuso':
        modelo = modelos[modelo_nombre]
        y_pred = predict_with_fcm(modelo, X)
    else:
        modelo = modelos[modelo_nombre]
        y_pred = modelo.predict(X)

    matriz = confusion_matrix(y_true, y_pred)
    exactitud = accuracy_score(y_true, y_pred)

    return render_template('resultado_batch.html',
                           matriz=matriz.tolist(),
                           exactitud=round(exactitud * 100, 2),
                           modelo_seleccionado=modelo_nombre)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

