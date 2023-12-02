import cv2
import numpy as np

# Inicializar el contador de vehículos
car_count = 0

# Colores para la visualización
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Paso 1: Obtener un objeto de captura de video
cap = cv2.VideoCapture("High4.mp4")

# Paso 2: Crear sustractor de fondo
subtract = cv2.createBackgroundSubtractorMOG2(
    history=5000, varThreshold=800, detectShadows=False
)


# Paso 3: Crear un kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))

# Posición de la línea para contar vehículos
linePos = 460
offset = 11  # Tolerancia en la detección de la línea

# Definir los límites de la zona de detección en el eje X
x_min = 200
x_max = 1200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Restringir la detección a la zona definida
    frame_zone = frame[:, x_min:x_max]

    # Paso 4: Convertir cada frame a escala de grises
    gray = cv2.cvtColor(frame_zone, cv2.COLOR_BGR2GRAY)

    # Paso 5: Aplicar GaussianBlur
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Paso 6: Aplicar la sustracción de fondo y operaciones morfológicas
    img_sub = subtract.apply(gray)
    dilation = cv2.dilate(img_sub, np.ones((15, 5), np.uint8))

    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    dilation2 = cv2.dilate(opening, np.ones((20, 20), np.uint8))
    opening2 = cv2.morphologyEx(dilation2, cv2.MORPH_CLOSE, kernel2)

    # Paso 7: Encontrar contornos
    contours, _ = cv2.findContours(opening2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Paso 8: Crear la línea para contar vehículos
    cv2.line(frame, (x_min, linePos), (x_max, linePos), BLUE, 2)

    # Paso 9: Iterar sobre cada contorno
    detect_car = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        valid_contour = (w >= 2) and (h >= 2)  # Establecer tus propios valores mínimos
        if not valid_contour:
            continue

        # Ajustar las coordenadas X para que coincidan con la imagen completa
        x += x_min
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        center_car = (int(x + w / 2), int(y + h / 2))
        detect_car.append(center_car)
        cv2.circle(frame, center_car, 4, RED, -1)

    # Paso 10: Contar vehículos
    for x, y in detect_car:
        if linePos - offset < y < linePos + offset:
            car_count += 1
            detect_car.remove((x, y))
            cv2.line(frame, (x_min, linePos), (x_max, linePos), ORANGE, 3)

    # Mostrar el número de vehículos detectados
    cv2.putText(
        frame,
        f"Numero de vehiculos: {car_count}",
        (50, 70),
        FONT,
        2,
        RED,
        3,
        cv2.LINE_AA,
    )

    # Mostrar el frame
    cv2.imshow("Vehicle Detection", frame)

    # Mostrar el frame después del filtrado
    cv2.imshow("Filtered Frame", opening2)
    cv2.imshow("Back", img_sub)

    # Romper el bucle con la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Liberar el video y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
