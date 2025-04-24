import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Ruta al archivo del modelo TFLite
MODEL_PATH = 'model/yolov5n-fp16.tflite'

# Cargar el modelo TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de las entradas y salidas del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Función para preprocesar la imagen
def preprocess_image(image, img_size=640):
    # Redimensionar la imagen
    image = cv2.resize(image, (img_size, img_size))

    # Convertir la imagen a float32
    image = np.array(image, dtype=np.float32)

    # Normalizar la imagen (dependiendo del modelo, ajusta según sea necesario)
    image = image / 255.0

    # Expande las dimensiones para que sea compatible con el modelo
    image = np.expand_dims(image, axis=0)

    return image

# Función para hacer la predicción
def predict(image):
    # Preprocesar la imagen
    input_data = preprocess_image(image)

    # Establecer el tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Ejecutar la inferencia
    interpreter.invoke()

    # Obtener las predicciones
    output_data = interpreter.get_tensor(output_details[0]['index'])	
    return output_data

def process_detections(output, image_shape, conf_threshold=0.25, iou_threshold=0.45):
    predictions = output[0]  # shape: (N, 85)

    boxes = predictions[:, :4]  # xywh
    objectness = predictions[:, 4]
    class_probs = predictions[:, 5:]  # shape: (N, 80)

    # Calcular scores combinando objectness y probabilidad de clase
    scores = objectness[:, None] * class_probs
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filtrar por umbral de confianza
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convertir de xywh (normalizado) a xyxy (píxeles)
    img_h, img_w = image_shape[:2]
    xyxy_boxes = []
    for box in boxes:
        x, y, w, h = box
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)
        xyxy_boxes.append([x1, y1, x2, y2])

    # Aplicar NMS usando OpenCV
    indices = cv2.dnn.NMSBoxes(xyxy_boxes, confidences.tolist(), conf_threshold, iou_threshold)
    indices = np.array(indices).flatten() if len(indices) > 0 else []

    final_boxes = [xyxy_boxes[i] for i in indices]
    final_scores = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]

    return final_boxes, final_scores, final_class_ids


# Capturar imágenes desde la webcam en tiempo real
cap = cv2.VideoCapture(0)  # 0 es el índice de la webcam, usa 1 si tienes varias cámaras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen")
        break

    # Realizar la predicción
    output = predict(frame)

    # Procesar las detecciones
    boxes, scores, class_ids = process_detections(output, frame.shape)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{COCO_CLASSES[class_id]}: {score:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar la imagen con las predicciones
    cv2.imshow("Webcam - YOLOv5 TFLite", frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
