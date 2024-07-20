import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

alph = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}

# Load the pre-trained model
model = torch.load('model_v1_trained.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# List to store characters
characters = []


def get_center_coordinates(frame, size=350):
    """Utility function to get the center square of the frame"""
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    x, y = center[0] - size // 2, center[1] - size // 2
    return x, y, size


def detect_gesture(frame, input_model):
    """ Detects the gesture in the center square of the frame with the trained model """
    # Assume the model expects a 28x28 input image
    x, y, size = get_center_coordinates(frame)
    hand_img = frame[y:y + size, x:x + size]
    hand_img = transform(Image.fromarray(hand_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = input_model(hand_img)
    gesture = output.argmax(dim=1).item()
    return gesture


def update_frame():
    """Updates the frame with the gesture prediction, if the hand is in the center, and displays it in the Tkinter window.
        Also displays the currently predicted gesture.
        Updates the global variable that stores the current gesture.
    """
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    # Draw green square in the center
    x, y, size = get_center_coordinates(frame)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)

    # Check if the hand is in the center square and predict gesture using the trained model
    hand_in_center = frame[y:y + size, x:x + size]
    gesture = detect_gesture(hand_in_center, model)

    # Display the gesture on the frame
    global gesture_to_alphabet
    gesture_to_alphabet = alph[gesture]
    gesture_text = f"Gesture: {gesture_to_alphabet}"
    cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Convert frame to Image for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    label.imgtk = img
    label.config(image=img)
    label.after(16, update_frame) # 16ms => 60 FPS

    # Update the character list display (Not needed, we do that with an event function)
    # char_list_label.config(text=" ".join(map(str, characters)))


def add_character(event):
    """Event function to add the currently predicted gesture to the list of characters. Triggered by SPACEBAR"""
    characters.append(gesture_to_alphabet)
    char_list_label.config(text="".join(map(str, characters)))


def quit_program(event):
    """Event function to quit the program. Triggered by ESCAPE"""
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


# Setup Tkinter
root = tk.Tk()
root.title("Sign Language Recognition")

label = Label(root)
label.pack()

char_list_label = Label(root, text="", font=('Helvetica', 18))
char_list_label.pack()

# Bind events
root.bind("<space>", add_character)
root.bind("<Escape>", quit_program)

# Get the camera feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Main loop
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
