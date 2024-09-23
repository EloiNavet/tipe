# Import librairies
import cv2
from PIL import Image
import numpy as np
from typing import List, Tuple


def analyze_brightness(img: np.ndarray) -> float:
    """
    Analyzes the brightness of a given image.

    Args:
        img (np.ndarray): Input image as a numpy array.

    Returns:
        float: The average brightness of the image (0-255 scale).
    """
    # Convert numpy array to PIL image and grayscale
    gray_img = Image.fromarray(img).convert("L")
    # Convert to numpy array and calculate the mean intensity
    brightness = np.mean(np.array(gray_img))
    return brightness

def image_to_vector(M: np.ndarray) -> np.ndarray:
    """
    Converts a square image matrix into a column vector by reshaping it.

    Args:
        M (np.ndarray): A square image matrix.

    Returns:
        np.ndarray: The reshaped matrix as a column vector of size (N, 1), where N = n * p (number of pixels).
    """
    n, p = M.shape
    return M.reshape((n * p, 1), order="F")


def resize_image(image_path: str, size: Tuple[int, int]):
    """
    Resizes the image to the given size and saves it.

    Args:
        image_path (str): Path to the image file.
        size (Tuple[int, int]): New size (width, height) for the image.
    """
    img = Image.open(image_path)
    resized_img = img.resize(size)
    resized_img.save(image_path)
    
def transform_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified path and converts it to grayscale.

    Args:
        image_path (str): The file path of the image.

    Returns:
        np.ndarray: The grayscale image as a numpy array.
    """
    img = Image.open(image_path)
    gray_image = img.convert("L")  # Convert to grayscale
    return np.array(gray_image)


def uniformize_brightness(
    M: np.ndarray, apply_uniformization: bool = False, order: int = 2
) -> np.ndarray:
    """
    Uniformizes the illumination of an image using the method described in:
    https://tel.archives-ouvertes.fr/tel-00623243/document (page 42).
    By default, it doesn't apply the uniformization unless specified.

    Args:
        M (np.ndarray): The input image matrix (grayscale).
        apply_uniformization (bool): Whether to apply illumination uniformization.
        order (int): The order of uniformization (default is 2).

    Returns:
        np.ndarray: The uniformized image matrix if `apply_uniformization` is True, otherwise the original matrix.
    """
    if not apply_uniformization:
        return M

    # Calculate the mean intensity of the image
    J = np.mean(M)
    hauteur, largeur = np.shape(M)
    Mbis = np.zeros((hauteur, largeur))

    # Calculate the row and column intensity lists
    hix_list, viy_list = [], []

    for x in range(largeur):
        hix = sum(M[y, x] ** (1 / order) for y in range(hauteur))
        hix_list.append((hix**order) / (hauteur * order))

    for y in range(hauteur):
        viy = sum(M[y, x] ** (1 / order) for x in range(largeur))
        viy_list.append((viy**order) / (largeur * order))

    # Calculate the uniformized matrix
    for x in range(hauteur):
        for y in range(largeur):
            Mbis[x, y] = hix_list[y] * viy_list[x] / J * M[x, y]

    # Scale back to the 0-255 range
    return (Mbis * (256 / np.max(Mbis))).astype(np.uint8)

def compute_mean(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the mean vector from a list of column vectors.

    Args:
        vectors (List[np.ndarray]): A list of column vectors (each a numpy array of shape (N, 1)).

    Returns:
        np.ndarray: The mean vector.
    """
    if not vectors:
        raise ValueError("The input list of vectors is empty.")
    
    return np.mean(vectors, axis=0)

def minimum(values: List[float]) -> Tuple[float, int]:
    """
    Finds the minimum value and its index in a list.

    Args:
        liste (List[float]): List of values.

    Returns:
        Tuple[float, int]: Minimum value and its index.
    """
    arr = np.array(values)
    idx = np.argmin(arr)
    return arr[idx], idx