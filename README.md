# TIPE: Face Recognition with Principal Component Analysis (PCA)

I made this project during preparatory classes in 2020 as part of my **TIPE** (Travaux d'Initiative Personnelle Encadrés). The main objective of the project was to implement an effective facial recognition system using **Principal Component Analysis (PCA)**. The project also explores **Linear Discriminant Analysis (LDA)** and compares to a naive matching algorithm as alternative recognition techniques.

## Project Overview

Facial recognition is an important field of biometric authentication, used in various applications like secure access, surveillance, and device unlocking. The goal of this project is to apply **PCA** to detect and recognize faces from a database of images. PCA allows us to reduce the dimensionality of facial images while retaining the most important features for recognition.

The project also integrates **LDA** for better discrimination in certain situations, addressing some of the limitations of PCA, such as sensitivity to lighting, pose, and facial expressions.

Note that today there are more advanced techniques like deep learning-based face recognition systems, but this project focuses on classical machine learning methods to be able to understand the underlying mathematical concepts and algorithms.

## Key Features

- **PCA-based facial recognition:** Implemented from scratch in Python using `numpy` and `opencv`.
- **LDA for class separation:** Adds robustness to the recognition system.
- **Naive face matching algorithm:** A simpler, pixel-by-pixel comparison method.
- **Webcam Integration:** Capture and analyze faces in real-time using a webcam.
- **Pygame Interface:** Visualizes known faces and the real-time camera feed.

## Project Structure

Here is the directory structure of the project:

```
├── data                    # Example face images
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
|   ├── ...
│   └── image.png # Image to be recognized
├── doc                     # Documentation and reports
│   ├── MCOT.pdf
│   ├── Presentation.pdf
│   ├── TIPE ACP_Analyse des données.xlsx
│   └── TIPE.pdf            # The short report written for the project
├── environment.yml          # Conda environment file with dependencies
├── README.md                # This README file
├── src                     # Source code
│   ├── acp.py              # PCA implementation
│   ├── adl.py              # LDA implementation
│   ├── dummy.py            # Naive face matching implementation
│   ├── main.py             # Main script for running the program
│   ├── utils.py            # Helper functions for image processing
└── └── visualization.py    # Functions for visualizing results
```

## Requirements

The project requires the following dependencies, which are listed in `environment.yml`. You can set up the environment using the following commands:

```bash
conda env create -f environment.yml
conda activate tipe
```

## How to Run

1. Clone the repository and set up the environment as described above.
2. Prepare your dataset of face images and place them in the `data/` folder.
3. Run the main program:

```bash
python src/main.py
```

### Modes:
- **Saving Faces Mode**: Capture and save face images to the database. You can enter this mode by pressing the **0** key.
- **Facial Recognition Mode**: Recognize faces in real-time from the webcam and compare them to the existing database of faces. You can switch to this mode by pressing the **1** key.

### Techniques:
- **PCA (Principal Component Analysis)**: This is the default method used for face recognition. It reduces the dimensionality of the image data, retaining only the most significant features (eigenfaces).
- **LDA (Linear Discriminant Analysis)**: Optional method for better accuracy, particularly useful when classifying faces of multiple people. This method maximizes the separation between different classes (people).
- **Naive Method**: A simple pixel-wise comparison of images. While not as robust as PCA or LDA, it serves as a basic baseline for comparison.

### Visualize Known Faces:
If you want to visualize the known faces stored in the database, the program can display them in a grid using **Pygame**. This feature allows you to see all the faces the program has already captured and is available when you set the `disp_faces` argument to `True`.

The main program will capture images through your webcam, detect faces, and either save them or attempt to recognize them based on the selected mode (`save` or `recognize`).

## Algorithms Overview

### 1. Naive Face Matching
This algorithm performs pixel-by-pixel comparisons between faces, calculating the difference for each pixel. While not as sophisticated as PCA or LDA, it provides a simpler baseline for face matching.

### 2. Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms facial images into a lower-dimensional space where the most significant features (i.e., eigenfaces) are retained. By projecting new images onto this space, we can compare and classify faces based on their proximity in this reduced space.

### 3. Linear Discriminant Analysis (LDA)
LDA is used to address some of the limitations of PCA by focusing on maximizing the separation between different classes (faces of different individuals), rather than just maximizing overall variance. This makes LDA more robust against variations in lighting and pose.

## Documentation

You can find detailed documentation about the project in the `doc/` folder, which includes:
- **TIPE.pdf**: The short report summarizing the facial recognition project and its objectives【38†source】.
- **MCOT.pdf & Presentation.pdf**: Additional presentations related to the project.
- **TIPE ACP_Analyse des données.xlsx**: An Excel file that contains data and analysis related to the PCA algorithm.

## Acknowledgements

This project was completed as part of my **TIPE** (2020 - 2021). I would like to thank my instructors for their guidance throughout the development of this project.

## References
- Eigenfaces: Turk, M. and Pentland, A. (1991) "Eigenfaces for Recognition"
- Fisherfaces (LDA): Belhumeur, P.N., Hespanha, J.P., and Kriegman, D.J. (1997) "Eigenfaces vs. Fisherfaces"