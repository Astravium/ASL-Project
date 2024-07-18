import cv2
import os
import re
import tkinter as tk
from tkinter import simpledialog

# Funzione per trovare il prossimo numero di cattura disponibile
def get_next_capture_number(directory, name, letter):
    existing_files = os.listdir(directory)
    max_number = 0
    pattern = re.compile(rf"{letter}_{name}_(\d+)\.jpg")
    for file in existing_files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    return max_number + 1

# Funzione per salvare l'immagine
def save_image(image, filename):
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, resized_image)
    print(f'Immagine salvata come {filename}')

# Funzione principale per la cattura delle immagini
def capture_images(name, letter):
    # Crea la cartella per il dataset se non esiste
    dataset_dir = os.path.join("Dataset", letter)
    os.makedirs(dataset_dir, exist_ok=True)

    # Numero di cattura
    capture_count = get_next_capture_number(dataset_dir, name, letter)

    # Dimensione dell'immagine ritagliata
    crop_size = 192

    # Apri la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam")
        return

    print("Premi 'c' per catturare e salvare l'immagine, 'q' per uscire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore: Impossibile acquisire l'immagine")
            break

        # Dimensioni del frame
        h, w, _ = frame.shape

        # Coordinate del centro del frame
        center_x, center_y = w // 2, h // 2

        # Coordinate del riquadro di cattura
        x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
        x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2

        # Disegna un riquadro verde al centro del frame (prima di catturare l'immagine)
        frame_with_rectangle = frame.copy()
        cv2.rectangle(frame_with_rectangle, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostra il frame con il riquadro
        cv2.imshow("Acquisizione", frame_with_rectangle)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Cattura l'immagine alla risoluzione normale senza il bordo verde
            capture_img = frame[y1:y2, x1:x2]
            filename = os.path.join(dataset_dir, f"{letter}_{name}_{capture_count}.jpg")
            save_image(capture_img, filename)
            capture_count += 1
        elif key == ord('q'):
            print("Uscita dal programma.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Funzione per avviare l'interfaccia utente
def start_ui():
    root = tk.Tk()
    root.withdraw()  # Nascondi la finestra principale

    # Chiedi il nome e la lettera utilizzando finestre di dialogo
    name = simpledialog.askstring("Input", "Inserisci il tuo nome:").upper()
    letter = simpledialog.askstring("Input", "Inserisci il carattere della LIS:").lower()

    # Avvia la cattura delle immagini
    capture_images(name, letter)

if __name__ == "__main__":
    start_ui()
