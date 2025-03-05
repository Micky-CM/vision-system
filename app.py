from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Variable global para almacenar la detección
person_detected = False

# -------------------- Cargar el modelo YOLOv3 --------------------
config_path = "models/yolov3.cfg"
weights_path = "models/yolov3.weights"
labels_path = "models/coco.names"

# Cargar etiquetas de COCO
LABELS = open(labels_path).read().strip().split("\n")

# Cargar la red neuronal de YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# Obtener nombres de las capas de salida
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def generate_frames():
    # Generador de frames que abre la cámara y aplica la detección de personas
    global person_detected
    video = cv2.VideoCapture(1) # 0 para webcam y 1 para cámara externa
    while True:
        ret, frame = video.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(ln)

        detected_in_frame = False  # Bandera para la detección en este frame

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # Si se detecta a una persona con confianza > 0.5
                if LABELS[classID] == "person" and confidence > 0.5:
                    detected_in_frame = True
                    box = detection[:4] * np.array([width, height, width, height])
                    (x_center, y_center, w, h) = box.astype("int")
                    x = int(x_center - (w / 2))
                    y = int(y_center - (h / 2))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Persona detectada", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Si se detectó en este frame, actualiza el flag global
        if detected_in_frame:
            person_detected = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Devuelve el stream de video con detección en tiempo real.
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_detection')
def check_detection():
    # Endpoint para consultar si se ha detectado una persona y reiniciar el flag.
    global person_detected
    detected = person_detected
    person_detected = False
    return jsonify({"detected": detected})

if __name__ == '__main__':
    app.run(debug=True)
