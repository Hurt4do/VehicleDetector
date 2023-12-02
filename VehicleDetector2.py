import cv2
import numpy as np

# Cargar los clasificadores en cascada para vehículos y peatones
car_cascade = cv2.CascadeClassifier("cars.xml")
pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Iniciar la captura de vídeo
video_src = "Bog.mp4"
cap = cv2.VideoCapture(video_src)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque Gaussiano
    # blur = cv2.GaussianBlur(gray, (2, 2), 0)

    # Dilatar la imagen
    dilated = cv2.dilate(gray, np.ones((2, 2)))

    # Aplicar cierre morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Mostrar la imagen después de las operaciones morfológicas
    cv2.imshow("Morphological Closing", closing)

    # Detectar vehículos y peatones
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)
    pedestrians = pedestrian_cascade.detectMultiScale(closing, 1.1, 1)

    # Dibujar rectángulos alrededor de los vehículos
    for x, y, w, h in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Dibujar rectángulos alrededor de los peatones
    for x, y, w, h in pedestrians:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Mostrar la imagen resultante con las detecciones
    cv2.imshow("Vehicle and Pedestrian Detection", img)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de vídeo y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
