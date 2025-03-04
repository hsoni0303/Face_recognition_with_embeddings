from extract_embeddings import extract_face_embeddings
import os

def registered_embeddings(registered_faces_folder):
    embeddings_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(registered_faces_folder):
        file_path = os.path.join(registered_faces_folder, filename)
        
        # Check if the file is an image
        if os.path.isfile(file_path) and (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
            embeddings = extract_face_embeddings(file_path)
            embeddings_dict[filename] = embeddings
    
    return embeddings_dict