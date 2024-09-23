from typing import List, Tuple

from utils import minimum

def find_face_dummy(V: List[float], visages: List[List[float]]) -> Tuple[str, float]:
    """
    Compares the pixels of the given face vector `V` with each face in the `visages` list
    (a list of column vectors) and returns the closest matching face along with a confidence factor.

    Args:
        V (list[float]): The face vector to compare.
        visages (list[List[float]]): A list of face vectors to compare against.

    Returns:
        Tuple[str, float]: The filename of the best matching face and the confidence factor (in percentage).
    """
    n = len(V)
    moys = []

    for visage in visages:
        # i is the pixel index, compute the absolute difference for each pixel
        dists = [abs(V[i] - visage[i]) for i in range(n)]
        moys.append(sum(dists[0]) / n)

    # Find the minimum and maximum average distances
    mini, maxi = min(moys), max(moys)

    # Compute the confidence factor (the closer to 100, the better the match)
    facteur = (1 - mini / maxi) * 100

    # Find the index of the closest matching face
    _, best_match_index = minimum(moys)

    # Construct the result text
    texte = f"Best match found: image{best_match_index}.png"

    return texte, facteur