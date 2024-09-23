import numpy as np
import matplotlib.pyplot as plt
from typing import List

from acp import build_A

def analyze_eigenvals_dist(visages: List[np.ndarray], moy: np.ndarray):
    """
    Analyzes and plots the eigenvalue distribution (scree plot) for PCA.

    Args:
        visages (list[np.ndarray]): List of face vectors.
        moy (np.ndarray): Mean vector of faces.
    """
    A = build_A(visages, moy)
    L = np.dot(np.transpose(A), A)
    valP, _ = np.linalg.eig(L)

    y = sorted(valP.tolist(), reverse=True)
    x = list(range(len(y)))

    plt.xlabel("N° Image")
    plt.ylabel("Valeur Propre")
    plt.title("Éboulement des valeurs propres")
    plt.bar(x, y)
    plt.show()