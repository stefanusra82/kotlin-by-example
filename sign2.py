import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ======== Setup Model dan MediaPipe =========
model_path = 'hand_gesture_model (4).h5'
model = load_model(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

number_labels = [
    "satu", "dua", "tiga", "empat", "lima",
    "enam", "tujuh", "delapan", "sembilan", "sepuluh",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]

# ======== Setup Tkinter Window =========
window = tk.Tk()
window.title("Hand Gesture Recognition")
window.geometry("980x540")
window.configure(bg="#111")
window.resizable(False, False)

# Frame kiri (panel kontrol)
left_frame = tk.Frame(window, width=300, height=540, bg="#222")
left_frame.grid(row=0, column=0, sticky="ns")
left_frame.grid_propagate(False)

# Judul besar dan tebal
title_label = tk.Label(left_frame, text="Hand Gesture Recognition", font=("Segoe UI", 16, "bold"),
                       fg="#00FFFF", bg="#222")
title_label.pack(pady=(30, 20))

# Status deteksi tangan
status_label = tk.Label(left_frame, text="Status: Tidak terdeteksi", font=("Segoe UI", 12),
                        fg="yellow", bg="#222")
status_label.pack(pady=10)

# Output gesture
prediction_label = tk.Label(left_frame, text="Output: -", font=("Segoe UI", 12, "bold"),
                            fg="lightgreen", bg="#222")
prediction_label.pack(pady=10)

# Tombol keluar
def exit_app():
    window.quit()

exit_button = tk.Button(left_frame, text="Selesai", command=exit_app,
                        font=("Segoe UI", 12, "bold"), bg="red", fg="white",
                        padx=30, pady=10)
exit_button.pack(pady=40)

# Frame kanan (kamera)
camera_label = tk.Label(window, bg="black", width=640, height=480)
camera_label.grid(row=0, column=1, padx=10, pady=20, sticky="n")

# ======== Buka Kamera =========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_label = "-"
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        hand_detected = True

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        if landmarks.shape[0] == 63:
            prediction = model.predict(landmarks.reshape(1, 63, 1), verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = prediction[0][class_id]
            predicted_label = f"{number_labels[class_id].upper()} ({confidence:.2f})"

    # Update UI
    prediction_label.config(text=f"Output: {predicted_label}")
    status_label.config(
        text="Status: Hand Detected" if hand_detected else "Status: Tidak terdeteksi",
        fg="lime" if hand_detected else "yellow"
    )

    img_pil = Image.fromarray(rgb_frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    window.after(10, update_frame)

# ======== Start Loop =========
update_frame()
window.mainloop()

# ======== Cleanup =========
cap.release()
hands.close()
cv2.destroyAllWindows()
