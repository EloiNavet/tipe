import cv2
import os
import numpy as np
import time
import concurrent.futures
from typing import List, Tuple

from utils import resize_image, compute_mean, image_to_vector, minimum, uniformize_brightness, transform_image

# Rappel : n = nombre de visages, N = nombre de pixels dans chaque image ==> taille des vecteurs colonne


def build_A(visages: List[np.ndarray], moy: np.ndarray) -> np.ndarray:
    """
    Constructs the matrix A by subtracting the mean vector from each face vector.

    Args:
        visages (list[np.ndarray]): List of face vectors.
        moy (np.ndarray): Mean vector of all face vectors.

    Returns:
        np.ndarray: The matrix A.
    """
    N = np.shape(visages[0])[0]
    A = np.zeros((N, 0))
    for vecteur in visages:
        A = np.concatenate((A, vecteur - moy), axis=1)
    return A


def build_eigenvectors(visages: List[np.ndarray], moy: np.ndarray) -> List[np.ndarray]:
    """
    Constructs the eigenvectors (Ei) of the covariance matrix.

    Args:
        visages (list[np.ndarray]): List of face vectors.
        moy (np.ndarray): Mean vector of all face vectors.

    Returns:
        List[np.ndarray]: List of eigenvectors.
    """
    A = build_A(visages, moy)
    L = np.dot(np.transpose(A), A)
    _, vectP = np.linalg.eig(L)
    return [np.dot(A, vectP[:, i]) for i in range(len(vectP))]


def project_face(
    V: np.ndarray, moy: np.ndarray, liste_ei: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Projects a face vector into the eigenface space.

    Args:
        V (np.ndarray): Face vector to project.
        moy (np.ndarray): Mean vector of faces.
        liste_ei (list[np.ndarray]): List of eigenvectors.

    Returns:
        List[np.ndarray]: List of projected coordinates in eigenface space.
    """
    V_Moy = V - moy
    return [np.dot(np.transpose(ei), V_Moy) for ei in liste_ei]


def build_projection_omega(
    V: np.ndarray, moy: np.ndarray, k: int, liste_ei: List[np.ndarray]
) -> np.ndarray:
    """
    Constructs the omega (projection) for a given face.

    Args:
        V (np.ndarray): Face vector to project.
        moy (np.ndarray): Mean vector of faces.
        k (int): Dimension of eigenface space.
        liste_ei (list[np.ndarray]): List of eigenvectors.

    Returns:
        np.ndarray: Projected coordinates (omega) of the face.
    """
    omega = project_face(V, moy, liste_ei)
    return np.array([omega[i][0] for i in range(k)])


def distance(
    visages: List[np.ndarray],
    V: np.ndarray,
    k: int,
    moy: np.ndarray,
    liste_ei: List[np.ndarray],
) -> List[float]:
    """
    Calculates the distance between a face and all faces in the dataset.

    Args:
        visages (list[np.ndarray]): List of face vectors.
        V (np.ndarray): Face vector to compare.
        k (int): Dimension of eigenface space.
        moy (np.ndarray): Mean vector of faces.
        liste_ei (list[np.ndarray]): List of eigenvectors.

    Returns:
        List[float]: List of distances between the given face and all other faces.
    """
    distances = []
    OmegaV = build_projection_omega(V, moy, k, liste_ei)
    for i in range(k):
        diff = OmegaV - build_projection_omega(visages[i], moy, k, liste_ei)
        d_i = np.linalg.norm(diff, 2)
        distances.append(d_i)
    return distances

def find_matching_face(
    V: np.ndarray,
    visages: List[np.ndarray],
    k: int,
    moy: np.ndarray,
    liste_ei: List[np.ndarray],
) -> Tuple[str, float]:
    """
    Finds the closest matching face to a given face vector.

    Args:
        V (np.ndarray): Face vector to compare.
        visages (list[np.ndarray]): List of known face vectors.
        k (int): Dimension of eigenface space.
        moy (np.ndarray): Mean vector of faces.
        liste_ei (list[np.ndarray]): List of eigenvectors.

    Returns:
        Tuple[str, float]: Path of the closest matching face and the confidence factor.
    """
    distances = distance(visages, V, k, moy, liste_ei)
    distance_mini, bonVisage = minimum(distances)
    texte = f"Best match found with `image{bonVisage}.png`"
    facteur = 100 * (1 - distance_mini / max(distances))
    return texte, facteur

def list_faces(image_paths: List[str]) -> List[np.ndarray]:
    """
    Processes a list of image paths, transforms each image into a grayscale matrix,
    normalizes its illumination, and converts it into a column vector.

    Args:
        image_paths (list[str]): List of file paths to face images.

    Returns:
        List[np.ndarray]: List of face vectors, each as a column vector.
    """
    return [
        image_to_vector(uniformize_brightness(transform_image(path)))
        for path in image_paths
    ]


def list_faces_threaded(image_paths: List[str]) -> List[np.ndarray]:
    """
    Processes a list of image paths using multithreading. Each image is transformed
    into a grayscale matrix, its illumination normalized, and converted into a column vector.

    Args:
        image_paths (list[str]): List of file paths to face images.

    Returns:
        List[np.ndarray]: List of face vectors, each as a column vector.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(uniformize_brightness, transform_image(path))
            for path in image_paths
        ]
    return [image_to_vector(future.result()) for future in futures]


def process_image_batch(face_paths: List[str], size: Tuple[int, int]):
    """
    Processes a batch of images by detecting and resizing faces.

    Args:
        face_paths (list[str]): List of image file paths to process.
        size (Tuple[int, int]): Desired size (width, height) for resizing.
    """
    for image_path in face_paths:
        # Detect face and resize
        detect_face(image_path)
        resize_image(image_path, size)

def detect_face_in_real_time(img: np.ndarray) -> np.ndarray:
    """
    Detects faces in a real-time video frame and draws rectangles around them.

    Args:
        img (np.ndarray): Frame from video as a numpy array.

    Returns:
        np.ndarray: The frame with detected faces outlined by rectangles.
    """
    # Use the built-in path to Haar cascades in OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    
    # Check if the cascade file loaded correctly
    if face_cascade.empty():
        raise IOError("Error loading cascade classifier")

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return img  # No faces detected, return original image

    # Find the largest face and draw a rectangle around it
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    img_with_face = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
    return img_with_face


def detect_face(image_path: str, show: bool = False) -> np.ndarray:
    """
    Detects the largest face in the image, crops it, and saves the result.

    Args:
        image_path (str): Path to the input image file.
        show (bool): If True, displays the detected face in a window.

    Returns:
        np.ndarray: The cropped face image, or the original image if no face is detected.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)

    if len(faces) == 0:
        # No faces detected, return the original image
        return img

    # Find the largest face by area (width * height)
    largest_face = max(faces, key=lambda face: face[2] * face[3])
    x, y, w, h = largest_face

    # Crop the largest detected face
    cropped_face = img[y : y + h, x : x + w]

    # Save the cropped face
    cv2.imwrite(image_path, cropped_face)

    if show:
        # Display the cropped face if `show` is True
        cv2.imshow("Detected Face", cropped_face)
        if cv2.waitKey() % 256 == 27:  # Close on 'ESC'
            cv2.destroyAllWindows()

    return cropped_face


def update_visage_text(data_path) -> (
    Tuple[List[str], List[np.ndarray], np.ndarray, List[np.ndarray]]
):
    """
    Updates the list of face images by reading the image files, transforming them into vectors,
    and computing the mean face vector and eigenvectors.

    Args:
        data_path (str): The path to the directory containing the face images.

    Returns:
        Tuple:
            - List of face image file paths.
            - List of face vectors (each as a column vector).
            - Mean face vector.
            - List of eigenvectors.
    """
    start_time = time.time()

    # Retrieve all PNG files in the current directory
    faces_text = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".png") and f.startswith("image") and len(f) >= 10]
    
    if not faces_text:
        raise FileNotFoundError("No face images found in the directory.")

    # Process images into vectors
    faces = list_faces(faces_text)

    # Compute the mean vector and eigenvectors
    mean_faces = compute_mean(faces)
    
    # Compute the eigenvectors
    ei_list = build_eigenvectors(faces, mean_faces)

    print(f"Faces list updated in {round(time.time() - start_time, 1)} seconds.")

    return faces_text, faces, mean_faces, ei_list

def detect_face_match(face_paths: List[str], image_path: str) -> str:
    """
    Detects and matches a face from an image against a set of known faces.

    Args:
        face_paths (list[str]): List of file paths for known face images.
        image_path (str): Path to the image that needs to be matched.

    Returns:
        str: The path of the closest matching face image.
    """
    # Generate list of face vectors for all known faces
    known_faces = [image_to_vector(transform_image(fp)) for fp in face_paths]
    # Convert the input image to vector
    input_vector = image_to_vector(transform_image(image_path))
    # Find and return the matching face
    return find_matching_face(input_vector, known_faces)

