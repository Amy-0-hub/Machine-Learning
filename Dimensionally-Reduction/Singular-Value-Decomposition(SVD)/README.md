# Handwritten Digit Classification Using SVD

This project implements a digit classification system using Singular Value Decomposition (SVD). The goal is to recognize handwritten digits by projecting them onto low-dimensional subspaces derived from known training data.


## Dataset

The data is sourced from the US Postal Service handwritten digit database:

- `trainInput.csv` – (256 x 1707): Training images (16×16 pixels, column-stacked into 256×1 vectors)
- `trainOutput.csv` – (1 x 1707): Labels for training images (digits 0–9)
- `testInput.csv` – (256 x 2007): Test images
- `testOutput.csv` – (1 x 2007): Labels for test images

Each column in the CSV files represents one image.


## Methodology
- Digit-wise Decomposition
- Projection + Reconstruction
- Classification
- Evaluation
   

## Results
- Overall Accuracy : 93.97%
- Best Classified Digit: (0) digit
- Worst Classified Digit: (5) digit


## Built with
1. pandas
2. numpy

