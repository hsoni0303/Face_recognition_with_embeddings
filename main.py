from capture import capture_image
from extract_embeddings import extract_face_embeddings
from recognition_score import calculate_cosine_similarity
from database_embeddings import registered_embeddings
import tkinter as tk
from tkinter import messagebox


if __name__ == "__main__":
    registered_users = "/Users/hemantsoni/Desktop/Face Recognition/Registered_Images"
    capture_path = "/Users/hemantsoni/Desktop/Face Recognition/captured_images"

    registered_data = registered_embeddings(registered_users)

    cap_img_path = capture_image(capture_path, "user")
    face_embedding = extract_face_embeddings(cap_img_path)

    if face_embedding == []:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Face Recognition", "Face Not Found!")
        exit(0)

    for k, v in registered_data.items():
        if calculate_cosine_similarity(v, face_embedding) > 0.7:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Face Recognition", "Match Found!")
            exit(0)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Face Recognition", "Match Not Found!")
    
    
