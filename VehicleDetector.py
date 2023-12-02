# ---------------------------------------------------------------
# Nombre del Archivo: VehicleDetector.py
# Autor: Juan Sebastian Hurtado
# Fecha de creación: 15/11/2023
# Última modificación: 20/11/2023
# Versión: 1.0
#
# Descripción:
# Este es un breve comentario describiendo lo que hace tu programa.
#
# Derechos de Autor: © Sebastian Hurtado, 2023
# Licencia: MIT License. Ver archivo LICENSE para detalles completos.
# ---------------------------------------------------------------


import cv2
import numpy as np

# Paso 1: Obtener un objeto de captura de video
cap = cv2.VideoCapture("1.mp4")

# Paso 2: Crear sustractor de fondo
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

# Paso 3: Crear un kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 4: Convertir cada frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Paso 5: Aplicar GaussianBlur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Paso 6: Aplicar la sustracción de fondo y operaciones morfológicas
    img_sub = subtract.apply(blur)
    dilation = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

    # Paso 7: Encontrar contornos
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Paso 9: Iterar sobre cada contorno
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        valid_contour = (w >= 2) and (h >= 2)  # Establecer tus propios valores mínimos
        if not valid_contour:
            continue

        # Dibujar un rectángulo alrededor de cada contorno detectado como vehículo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow("Vehicle Detection", frame)

    # Romper el bucle con la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Liberar el video y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
