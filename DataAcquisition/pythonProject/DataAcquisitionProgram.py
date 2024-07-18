import cv2
import os

# Richiedi il nome dell'utente e l'etichetta del carattere
name = input("Inserisci il tuo nome: ").upper()
letter = input("Inserisci il carattere della LIS: ").lower()

# Crea la cartella per il dataset se non esiste
dataset_dir = os.path.join("Dataset", letter)
os.makedirs(dataset_dir, exist_ok=True)

# Numero di cattura
capture_count = 0

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
        capture_count += 1
        file_name = os.path.join(dataset_dir, f"{letter}_{name}_{capture_count}.jpg")
        save_image(crop_img, file_name)
    elif key == ord('q'):
        print("Uscita dal programma.")
        break

cap.release()
cv2.destroyAllWindows()
