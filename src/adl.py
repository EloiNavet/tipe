# Import librairies
import os
from PIL import Image
import numpy as np
from typing import List, Tuple

from utils import image_to_vector

# We assume (for now) that the list of face paths (faces_text)
# is in the form "image_i_j.png" where i corresponds to person i,
# and j to the id of that person’s image.


def compute_mean_adl(faces: List[List[np.ndarray]]) -> np.ndarray:
    """
    Calculates the mean face vector for ADL.

    Args:
        faces (List[List[np.ndarray]]): A list of face vectors, grouped by person.

    Returns:
        np.ndarray: The mean face vector for all faces.
    """
    return np.mean([face for group in faces for face in group], axis=0)


def transform_image_adl(path: str) -> np.ndarray:
    """
    Resizes an image to 100x100 and converts it to grayscale for ADL.

    Args:
        path (str): The path to the image.

    Returns:
        np.ndarray: Grayscale image as a numpy array.
    """
    img = Image.open(path)
    img = img.resize((100, 100))
    image = img.convert("L")  # Convert to grayscale
    return np.array(image)


def build_faces_adl(face_path: List[str]) -> List[List[np.ndarray]]:
    """
    Constructs the list of faces for ADL, where each sublist corresponds to a person.

    Args:
        face_paths (list[str]): List of image paths.

    Returns:
        List[List[np.ndarray]]: A nested list of face vectors grouped by person.
    """
    face_index = 0
    res = [[]]
    for face in face_path:
        components = face.split("_")
        if int(components[1]) != face_index:
            face_index += 1
            res.append([])
        res[-1].append(image_to_vector(transform_image_adl(face)))
    return res


def compute_psi(faces: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Calculates the mean vector for each class (i.e., for each person).

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors, grouped by person.

    Returns:
        List[np.ndarray]: A list of mean vectors for each class.
    """
    return [(1 / len(faces[i]) * sum(faces[i])) for i in range(len(faces))]


def compute_phi(
    faces: List[List[np.ndarray]], psi_list: List[np.ndarray]
) -> List[List[np.ndarray]]:
    """
    Calculates the normalized faces for each class by subtracting the mean vector (psi).

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors.
        psi_list (list[np.ndarray]): List of mean vectors for each class.

    Returns:
        List[List[np.ndarray]]: A nested list of normalized face vectors.
    """
    return [[faces[i][k] - psi_list[i] for k in range(len(faces[i]))] for i in range(len(faces))]


def calculate_within_class_dispersion(
    faces: List[List[np.ndarray]], psi_list: List[np.ndarray]
) -> np.ndarray:
    """
    Calculates the within-class dispersion matrix (S_W) for ADL.

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors grouped by class.
        psi_list (list[np.ndarray]): A list of mean vectors for each class.

    Returns:
        np.ndarray: The within-class dispersion matrix.
    """
    c = len(faces)
    size = np.shape(faces[0][0])[0]
    S_W = np.zeros((size, size))
    for i in range(c):
        for face in faces[i]:
            column = face - psi_list[i]
            S_W += np.dot(column, np.transpose(column))
    return S_W


def calculate_between_class_dispersion(
    faces: List[List[np.ndarray]], mean_adl: np.ndarray, psi_list: List[np.ndarray]
) -> np.ndarray:
    """
    Calculates the between-class dispersion matrix (S_B) for ADL.

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors grouped by class.
        mean_adl (np.ndarray): The overall mean face vector.
        psi_list (list[np.ndarray]): A list of mean vectors for each class.

    Returns:
        np.ndarray: The between-class dispersion matrix.
    """
    size = np.shape(faces[0][0])[0]
    S_B = np.zeros((size, size))
    for i in range(len(faces)):
        column = psi_list[i] - mean_adl
        S_B += np.dot(column, np.transpose(column))
    return S_B


def calculate_total_dispersion(
    faces: List[List[np.ndarray]], mean_adl: np.ndarray
) -> np.ndarray:
    """
    Calculates the total dispersion matrix (S_T) for ADL.

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors grouped by class.
        mean_adl (np.ndarray): The overall mean face vector.

    Returns:
        np.ndarray: The total dispersion matrix.
    """
    size = np.shape(faces[0][0])[0]
    S_T = np.zeros((size, size))
    for i in range(len(faces)):
        for k in range(len(faces[i])):
            column = faces[i][k] - mean_adl
            S_T += np.dot(column, np.transpose(column))
    return S_T


def find_w(faces: List[List[np.ndarray]], S_W: np.ndarray, S_B: np.ndarray):
    """
    Finds the transformation matrix W for ADL.

    If the rank of the within-class dispersion matrix (S_W) is smaller than
    the number of classes (people), then the matrix is non-invertible. In such cases,
    a dimensionality reduction using PCA is typically applied before ADL.

    Args:
        faces (list[List[np.ndarray]]): A nested list of face vectors grouped by class.
        S_W (np.ndarray): The within-class dispersion matrix.
        S_B (np.ndarray): The between-class dispersion matrix.
    """
    if np.linalg.matrix_rank(S_W) <= len(faces):
        print("Within-class dispersion matrix rank too small.")
        return

    # Calculate W using eigenvalues and eigenvectors
    matrix = np.dot(np.linalg.inv(S_W), S_B)
    valP, vectP = np.linalg.eig(matrix)
    return vectP


def update_faces_text_adl(data_path) -> Tuple[
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Updates the ADL face dataset by computing the mean vector, class means (psi),
    normalized faces (phi), and within-class and between-class dispersion matrices.
    
    Args:
        data_path (str): The path to the directory containing the face images.

    Returns:
        Tuple: The overall mean vector, list of psi vectors, phi vectors,
               within-class dispersion matrix, between-class dispersion matrix, and total dispersion matrix.
    """
    faces_path_adl = [f for f in os.listdir(data_path) if f.endswith('.png')]
    faces_adl = build_faces_adl(faces_path_adl)
    mean_adl = compute_mean_adl(faces_adl)
    psi_list = compute_psi(faces_adl)
    phi_list = compute_phi(faces_adl, psi_list)
    S_W = calculate_within_class_dispersion(faces_adl, psi_list)
    S_B = calculate_between_class_dispersion(faces_adl, mean_adl, psi_list)
    S_T = calculate_total_dispersion(faces_adl, mean_adl)

    print("Face list for Linear Discriminant Analysis (ADL) updated.")
    return mean_adl, psi_list, phi_list, S_W, S_B, S_T


# moyADL, listePsi, listePhi, dispIntraClasse, dispInterClasse, dispersion = update_face_text_adl("C:/Users/Eloi/OneDrive/Desktop/TIPE/Mes images/ADL")
