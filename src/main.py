# Import librairies
import cv2
import os
from PIL import Image
import numpy as np
import pygame
import time
from typing import List
import argparse

from acp import find_matching_face, detect_face_in_real_time, update_visage_text, resize_image, uniformize_brightness, detect_face, transform_image, image_to_vector
from dummy import find_face_dummy
from utils import analyze_brightness

DATABASE_PATH = "../data"
SIZE = (220, 250)  # width, height: best size for face recognition and ratio quality/size

def main(faces_text: List[str],
                  faces: List[np.ndarray],
                  mean_faces: np.ndarray,
                  ei_list: List[np.ndarray],
                  mode: int, 
                  tech: str = "ACP", 
                  disp_faces: bool = False):
    """
    Captures an image using a webcam, processes it for face recognition or saves it. Optionally displays known faces in a pygame window.

    Args:
        faces_text (List[str]): List of paths to known face images.
        faces (List[np.ndarray]): List of known face vectors.
        mean_faces (np.ndarray): Mean face vector for comparison.
        ei_list (List[np.ndarray]): List of eigenvectors for face recognition.
        mode (int): 0 for saving faces, 1 for facial recognition.
        tech (str): Recognition technique ('ACP', 'naif', 'ADL'). Defaults to 'ACP'.
        disp_faces (bool): Whether to display known faces using pygame. Defaults to False.

    """
    # Ensure correct mode and recognition technique
    assert tech in ["ACP", "naif", "ADL"], "`tech` must be 'ACP', 'naif', or 'ADL'"
    assert mode in [0, 1], "`mode` must be 0 or 1"

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face")

    img_counter = len(faces_text)
    count = 0

    ########## PYGAME DISPLAY ##########
    if disp_faces:
        pygame.init()
        font = pygame.font.SysFont(None, 25)
        white = (255, 255, 255)
        black = (0, 0, 0)

        window_size = (1200, 800)
        img_per_row = 8

        # Load first image to calculate the aspect ratio
        img = cv2.imread(faces_text[0])
        h, w, d = img.shape
        aspect_ratio = h / w

        img_width = window_size[0] // img_per_row - 10
        img_height = int(img_width * aspect_ratio)

        # Create pygame window
        window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Face Recognition")

        def display_message(msg: str, color, position):
            """Displays a message on the pygame window."""
            text = font.render(msg, True, color)
            window.blit(text, position)

        def display_image(path: str, position, size):
            """Displays an image on the pygame window."""
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, size)
            window.blit(img, position)

        def display_known_faces():
            """Displays all known faces on the pygame window."""
            window.fill(white)
            mouse_pos = pygame.mouse.get_pos()

            # Detect mouse click on the screen
            if pygame.mouse.get_pressed()[0] and 110 < mouse_pos[0] < 190 and 140 < mouse_pos[1] < 180:
                print("Mouse clicked")

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()

            # Display all known faces
            row_index = 0
            for i, face_path in enumerate(faces_text):
                if i % img_per_row == 0 and i > 0:
                    row_index += 1
                x = 10 + (img_width + 10) * (i % img_per_row)
                y = (3 * 10 + img_height) * row_index + 10
                display_image(face_path, (x, y), (img_width, img_height))
                display_message(f"Image {i+1}", black, (x + 10, y + img_height + 10))

            pygame.display.update()

        display_known_faces()

    #################### MAIN LOOP ####################

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect face in real-time video feed
        framed_img = detect_face_in_real_time(frame)
        cv2.imshow("Face", framed_img)

        # Analyze brightness every 20 frames
        if count % 20 == 0:
            brightness = analyze_brightness(framed_img)
            mean_faces = brightness / 256 * 100
            if mean_faces >= 60:
                print("Brightness is too high.")
            elif mean_faces <= 40:
                print("Brightness is too low.")

        key = cv2.waitKey(1)

        # Exit on ESC
        if key % 256 == 27:
            break
        # Switch to face saving mode
        elif key % 256 == 224:
            mode = 0
            print("Face saving mode activated.")
        # Switch to face recognition mode
        elif key % 256 == 38:
            mode = 1
            print("Face recognition mode activated.")
        # Capture image on SPACE
        elif key % 256 == 32:
            img_name = os.path.join(DATABASE_PATH, f"image{img_counter if mode == 0 else ''}.png")
            cv2.imwrite(img_name, frame)
            print(f"Image '{img_name}' saved!")

            detect_face(img_name)
            im = Image.open(faces_text[0])
            resize_image(img_name, size=(im.width, im.height))

            # Update faces list and counter if in saving mode
            if mode == 0:
                faces_text, faces, mean_faces, ei_list = update_visage_text(DATABASE_PATH)
                img_counter += 1
                display_known_faces()
            # Perform face recognition if in recognition mode
            if mode == 1:
                t0 = time.time()
                V = image_to_vector(uniformize_brightness(transform_image(img_name)))
                if tech == "ACP":
                    k = len(faces)  # Set dimension of projection space
                    # IMPORTANT: k can be set to a lower value thanks to the theory described in docs. Ideally, k can be set to len//10 and this will speed up the process with a minimal loss
                    text, ratio = find_matching_face(V, faces, k, mean_faces, ei_list)
                elif tech == "naif":
                    text, ratio = find_face_dummy(V, faces)
                elif tech == "ADL":
                    raise NotImplementedError("ADL not implemented yet.")

                print(f"{text}, with a confidence of {round(ratio, 2)}%, in {round(time.time() - t0, 3)}s.")

        count += 1

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    if disp_faces:
        pygame.quit()



if __name__ == "__main__":
    # Initialize the list of face images and related data
    faces_text, faces, mean_faces, ei_list = update_visage_text(DATABASE_PATH)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tech", type=str, default="ACP", choices=["ACP", "naif", "ADL"])
    parser.add_argument("--disp_faces", type=bool, default=True)

    args = parser.parse_args()
    
    main(faces_text, faces, mean_faces, ei_list, args.mode, args.tech, args.disp_faces)
