import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load pre-trained Facenet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

def extract_face_embeddings(image_path):
  # Load the image
  img = Image.open(image_path)

  # Detect faces in the image
  boxes, _ = mtcnn.detect(img)

  # Extract embeddings for each detected face
  embeddings = []
  if boxes is not None:
    for box in boxes:
      # Crop the face
      face = img.crop(box)

      # Preprocess the face (resize and normalize)
      face = face.resize((160, 160))
      face = np.array(face).astype(np.float32) / 255.0
      face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
      face = np.transpose(face, (2, 0, 1))  # Convert to (C, H, W)

      # Convert to torch tensor and add batch dimension
      face = torch.tensor(face).unsqueeze(0)

      # Extract embedding
      with torch.no_grad():
          embedding = model(face)

      embeddings.append(embedding.numpy().flatten())

  return embeddings
