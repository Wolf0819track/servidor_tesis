from flask import Flask, request
import os
import cv2
import tempfile
from werkzeug.utils import secure_filename
from flask import send_file
import shutil
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from flask import Flask, jsonify
from spellchecker import SpellChecker
from itertools import groupby
from gtts import gTTS
import base64
import spacy

app = Flask(__name__)

# Ruta donde se guardarán los fotogramas recibidos
UPLOAD_FOLDER = 'fotogramas'
UPLOAD_FOLDER = tempfile.mkdtemp()
print("📁 Carpeta temporal:", UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
nlp = spacy.load("es_core_news_sm")
spell = SpellChecker(language='es') 

# Ruta raíz para comprobar que el servidor responde
@app.route("/")
def home():
    return "Servidor en línea 🚀"

# Aquí irían tus otras rutas
@app.route("/predict", methods=["POST"])
def predict():
    # lógica de predicción
    return "Predicción"

@app.route('/recibir', methods=['POST'])
def recibir_imagenes():
    files = request.files.getlist('imagenes')  # ← Coincide con el nombre usado en Unity

    if not files:
        return 'No se recibieron archivos', 400 
    temp_dir = '/tmp/uploads'
    temp_dir_2='C:\\Users\\crist\\Documents\\UMNG-2025\\Tesis\\frames unity'
    os.makedirs(temp_dir, exist_ok=True)
    limpiar_directorio(temp_dir);
    for idx, file in enumerate(files):
        if file and file.filename != '':
            # Asegurar nombre de archivo válido
            filename = secure_filename(file.filename)
            if not filename.lower().endswith('.jpg'):
                filename += '.jpg'
            
            # Guardar en ambos directorios
            for directory in [temp_dir, temp_dir_2]:
                file_path = os.path.join(directory, f"{idx}_{filename}")
                try:
                    file.seek(0)  # Rebobinar el archivo por si acaso
                    file.save(file_path)
                except Exception as e:
                    print(f"Error guardando {file_path}: {str(e)}")
    output_path = os.path.join(temp_dir, 'output.mp4')
    palabra=analizar_frames_en_directorio(temp_dir);
    resultado_audio = get_audio( );
    return jsonify({
        "mensaje": f'{len(files)} imágenes recibidas correctamente ✅',
        "audio": resultado_audio,
        "palabra": palabra
    }), 200

def analizar_frames_en_directorio(directorio_frames):
    model = tf.keras.models.load_model('modelo_X_5.h5')
    scaler = joblib.load('escalador_5.pkl')
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

    mp_hands = mp.solutions.hands
    letras_detectadas = []
    frame_counter = 0
    pred_interval = 2

    frames = sorted(
    [f for f in os.listdir(directorio_frames) if f.endswith('.jpg')],
    key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extrae números y convierte a int
    )
    print(f"\n🔍 Iniciando análisis de {len(frames)} frames...")

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

        for nombre_frame in frames:
            path = os.path.join(directorio_frames, nombre_frame)
            frame = cv2.imread(path)
            if frame is None:
                print(f"⚠️ Error al leer el frame: {nombre_frame}")
                continue

            frame_counter += 1
            if frame_counter % pred_interval != 0:
                continue
            print(f"\n📂 Procesando frame: {nombre_frame}")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                coords = [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]

                X_pred = scaler.transform([coords])
                probs = model.predict(X_pred, verbose=0)[0]
                idx = np.argmax(probs)
                confidence = probs[idx]

                if confidence > 0.90:
                    letra = class_names[idx]
                    letras_detectadas.append(letra)
                    #print(f"Letra detectada: {letra} ({confidence:.2f})")
                    print(f"   👉 Letra detectada: {letra} ({confidence:.2%}) en {nombre_frame}")  # <-- Mejorado
                    letras_detectadas = list(limpiar_intermedias(letras_detectadas))

    salida = autocorregir_palabra(letras_detectadas)
    print("\nCorrector_1:", salida)  # salida: hola

    palabra_final = autocorregir_letras(letras_detectadas)
    print(f"\n📝 Coorector_2: {palabra_final}")

    hablar(salida);   
    return salida

def convertir_imagenes_a_video(directorio, salida):
    imagenes = sorted([img for img in os.listdir(directorio) if img.endswith(".jpg")])
    if not imagenes:
        return
    ruta_imagen = os.path.join(directorio, imagenes[0])
    frame = cv2.imread(ruta_imagen)
    alto, ancho, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(salida, fourcc, 30, (ancho, alto))

    for imagen in imagenes:
        ruta = os.path.join(directorio, imagen)
        frame = cv2.imread(ruta)
        out.write(frame)

    out.release()
 
# === Postprocesamiento: autocorregir repeticiones ===
def autocorregir_letras(letras):
    palabra = []
    for i, l in enumerate(letras):
        if i == 0 or l != letras[i - 1]:
            palabra.append(l)
    return "".join(palabra)

def limpiar_intermedias(letras, min_repeticiones=3):
    resultado = []
    i = 0
    while i < len(letras):
        actual = letras[i]
        j = i
        while j < len(letras) and letras[j] == actual:
            j += 1
        grupo_len = j - i
        if grupo_len >= min_repeticiones:
            resultado.extend([actual] * grupo_len)
            i = j
        else:
            letra_anterior = letras[i - 1] if i - 1 >= 0 else None
            letra_siguiente = letras[j] if j < len(letras) else None

            if (
                letra_anterior == letra_siguiente
                and letra_anterior is not None
            ):
                i = j
            else:
                resultado.extend(letras[i:j])
                i = j

    return ''.join(resultado)

def limpiar_repeticiones(texto):
    texto=''.join(texto)
    resultado = ''
    for letra, grupo in groupby(texto):
        repeticiones = len(list(grupo))
        if repeticiones >= 3:
            resultado += letra  # Solo guarda letras que se repiten al menos 3 veces
    print('palabra_limpia:'+resultado)
    return resultado

def autocorregir_palabra(palabra):
    palabra_limpia = limpiar_repeticiones(palabra)
    doc = nlp(palabra_limpia)
    for token in doc:
        if token.ent_type_ == "PER":
            return palabra_limpia  
    correccion = spell.correction(palabra_limpia)
    return correccion if correccion else palabra_limpia

def hablar(texto, idioma='es'):
    if not texto.strip():
        print("❌ Texto vacío, no se puede convertir a voz.")
        return None
    temp_dir = '/tmp/uploads'
    # Crear archivo temporal
    archivo = os.path.join(temp_dir, "voz.mp3")
    # Generar y guardar audio
    tts = gTTS(text=texto, lang=idioma)
    tts.save(archivo)
    print(f"✅ Audio guardado en: {archivo}")



def get_audio( ):
    # Tu lógica para generar o cargar el audio basado en 'palabra'
    with open("/tmp/uploads/voz.mp3", "rb") as f:
        audio_binario = f.read()
    audio_base64 = base64.b64encode(audio_binario).decode("utf-8")
    return {
        "audio": audio_base64
    }


def limpiar_directorio(directorio):
    for archivo in os.listdir(directorio):
        archivo_path = os.path.join(directorio, archivo)
        try:
            if os.path.isfile(archivo_path):
                os.unlink(archivo_path)
            elif os.path.isdir(archivo_path):
                shutil.rmtree(archivo_path)
        except Exception as e:
            print(f"❌ Error al eliminar {archivo_path}: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
