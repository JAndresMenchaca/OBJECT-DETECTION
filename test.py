import cv2
import numpy as np

# Cargar los archivos de configuración y pesos del modelo YOLOv3
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Cargar las clases del conjunto de datos COCO
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generar colores aleatorios para cada clase
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer el cuadro de la cámara
    ret, frame = cap.read()

    # Obtener las dimensiones del cuadro
    height, width, _ = frame.shape

    # Preprocesar el cuadro para que coincida con el formato de entrada del modelo
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Establecer la entrada del modelo
    net.setInput(blob)

    # Obtener las salidas de las capas de detección
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(outputlayers)


    # Inicializar listas para las cajas delimitadoras, confianzas y clases detectadas
    boxes = []
    confidences = []
    class_ids = []

    # Recorrer las salidas de las capas de detección
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Umbral de confianza

                # Obtener las coordenadas de la caja delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calcular las coordenadas de la esquina superior izquierda de la caja delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Actualizar las listas de cajas, confianzas y clases detectadas
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar la supresión no máxima para eliminar las detecciones superpuestas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar las cajas delimitadoras y etiquetas en el cuadro
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 0.6, color, 2)

    # Mostrar el cuadro resultante
    cv2.imshow("Object Detection", frame)

    # Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
