import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Ruta al archivo del modelo TFLite
MODEL_PATH = 'model/yolov5n-fp16.tflite'

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
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


# Capturar imágenes desde la webcam en tiempo real
cap = cv2.VideoCapture(0)  # 0 es el índice de la webcam, usa 1 si tienes varias cámaras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen")
        break

    # Realizar la predicción
    output = predict(frame)

    # Aquí podrías procesar la salida para extraer las cajas delimitadoras (bounding boxes)
    # Por ejemplo, si el modelo devuelve los bounding boxes y las clases predichas, puedes dibujarlas:

    # Asumiendo que la salida contiene boxes y clases
    for detection in output[0]:
        # Extrae las coordenadas de la caja y las clases
        # Esto depende de cómo se haya entrenado el modelo, normalmente se devuelve [x, y, w, h, score, class_id]
        x, y, w, h, score, class_id = detection[:6]

        # Si el puntaje de la predicción es mayor que un umbral
        if score > 0.5:
            # Dibujar la caja delimitadora en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            cv2.putText(frame, f'Class: {int(class_id)} Score: {score:.2f}', (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar la imagen con las predicciones
    cv2.imshow("Webcam - YOLOv5 TFLite", frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
