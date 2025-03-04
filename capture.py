import cv2
import os
import sys
import time  # Import the time module for delay

def capture_image(image_folder, file_name):
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    # Wait for 5 seconds
    time.sleep(5)
    
    # Capture a picture
    ret, frame = camera.read()
    
    # Release the camera
    camera.release()
    
    if ret:
        # Define the folder to store the images
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Construct the file path
        image_path = os.path.join(image_folder, f"{file_name}.jpg")

        # Save the captured image to file
        cv2.imwrite(image_path, frame)

        return image_path

    else:
        print("Failed to capture image.")
        sys.exit(1)