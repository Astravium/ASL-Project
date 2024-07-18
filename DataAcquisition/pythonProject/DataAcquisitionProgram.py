import cv2
import os
import re

# Richiedi il nome dell'utente e l'etichetta del carattere
name = input("Inserisci il tuo nome: ").upper()
letter = input("Inserisci il carattere della LIS: ").lower()

# Crea la cartella per il dataset se non esiste
dataset_dir = os.path.join("Dataset", letter)
os.makedirs(dataset_dir, exist_ok=True)

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

# Numero di cattura
capture_count = get_next_capture_number(dataset_dir, name, letter)

# Dimensione dell'immagine ritagliata
crop_size = 64

def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f'Immagine salvata come {filename}')

# Apri la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Errore: Impossibile aprire la webcam")
    exit()

print("Premi 'c' per catturare e salvare l'immagine, 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore: Impossibile acquisire l'immagine")
        break

    # Ritaglia un quadrato di 64x64 pixel dal centro dell'immagine
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
    x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2
    crop_img = frame[y1:y2, x1:x2]

    # Mostra l'immagine ritagliata
    cv2.imshow("Acquisizione", crop_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = os.path.join(dataset_dir, f"{letter}_{name}_{capture_count}.jpg")
        save_image(crop_img, filename)
        capture_count += 1
    elif key == ord('q'):
        print("Uscita dal programma.")
        break

cap.release()
cv2.destroyAllWindows()
