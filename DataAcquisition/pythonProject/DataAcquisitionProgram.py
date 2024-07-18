import cv2
import os
import re
import tkinter as tk
from tkinter import messagebox

# Define global variables
window_closed = False
user_name = ""
user_letter = ""


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

    window_name = "Acquisizione"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # Crea una finestra non ridimensionabile

    # Utilizza tkinter per ottenere le dimensioni dello schermo
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di tkinter
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Centra la finestra di acquisizione sullo schermo
    window_width = 640  # Larghezza della finestra
    window_height = 480  # Altezza della finestra
    pos_x = int((screen_width - window_width) / 2)
    pos_y = int((screen_height - window_height) / 2)
    cv2.moveWindow(window_name, pos_x, pos_y)

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
        cv2.imshow(window_name, frame_with_rectangle)

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

        # Verifica se la finestra è stata chiusa
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
                print("Finestra chiusa")
                break
        except cv2.error:
            print("Finestra chiusa con eccezione")
            break

    cap.release()
    cv2.destroyAllWindows()


# Funzione per creare la finestra di dialogo personalizzata
def get_user_input():
    def on_submit(event=None):
        global user_name, user_letter
        user_name = name_entry.get().upper()
        user_letter = letter_entry.get().lower()

        # Controlli di validità
        if not user_name:
            messagebox.showerror("Errore", "Il nome non può essere vuoto.")
            return
        if user_name.isdigit():
            messagebox.showerror("Errore", "Il nome non può essere un numero.")
            return
        if not user_letter or len(user_letter) != 1:
            messagebox.showerror("Errore", "Il carattere deve essere esattamente uno.")
            return
        if user_letter in ['g', 's', 'j', 'z']:
            messagebox.showerror("Errore", "Il carattere inserito non è consentito.\n"
                                           "I caratteri G, S, J, Z sono esclusi.")
            return
        if user_letter.isdigit():
            messagebox.showerror("Errore", "Il carattere non può essere un numero.")
            return

        dialog.destroy()

    def on_close():
        global window_closed
        window_closed = True
        dialog.destroy()

    dialog = tk.Tk()
    dialog.title("Inserisci le informazioni")

    # Centra la finestra di dialogo sullo schermo
    window_width = 400
    window_height = 300
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    dialog.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    tk.Label(dialog, text="Nome:", font=("Arial", 14)).pack(pady=10)
    name_entry = tk.Entry(dialog, font=("Arial", 14))
    name_entry.pack(pady=10)

    tk.Label(dialog, text="Carattere della LIS:", font=("Arial", 14)).pack(pady=10)
    letter_entry = tk.Entry(dialog, font=("Arial", 14))
    letter_entry.pack(pady=10)

    submit_button = tk.Button(dialog, text="Submit", font=("Arial", 14), command=on_submit)
    submit_button.pack(pady=20)

    dialog.bind('<Return>', on_submit)
    dialog.protocol("WM_DELETE_WINDOW", on_close)

    dialog.mainloop()


# Funzione per avviare l'interfaccia utente
def start_ui():
    global window_closed
    window_closed = False
    get_user_input()
    if not window_closed:
        capture_images(user_name, user_letter)


if __name__ == "__main__":
    start_ui()
