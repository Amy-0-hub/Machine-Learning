# Landsat Image PCA Pipeline Project

This project performs Principal Component Analysis (PCA) on a stack of 5 satellite images from Landsat data, each representing a different wavelength band. The goal is to reduce these multi-band images into a single-channel image while preserving the most important variance.


## Objectives

1. Load and plot all 5 Landsat images.
2. Set up a PCA pipeline with linear scaling between 0 and 1.
3. Combine all 5 bands into a single image using PCA.
4. Analyze PCA components and report:
   - Number of samples and features
   - Principal directions (eigenvectors)
   - Explained variance ratio
   - Value of the last pixel in the new image
5. Display the resulting PCA-reduced image.



## Methodology
- Image Loading & Visualization
- Data Preprocessing + PCA Pipeline
 - Images were flattened and stacked into a data matrix of shape.
 - Used `MinMaxScaler` to scale values to the [0, 1] range.
 - Applied `PCA(n_components=1)` from `sklearn` to reduce to a single channel.


## Built with
1. numpy
2. pandas
3. statsmodels
4. sklearn
5. plotly

### PCA Analysis

- Extracted and printed PCA components (directions).
- Reported the explained variance ratio of the first principal component.
- Computed and printed the value of the **last pixel** in the final PCA image.


## Results Summary

- **Number of Samples**: 57772701
- **Number of Features**: 5
- **Explained Variance Ratio** (1st component):`99.2%`
- **PCA Directions**:  
  ```text
  [[-0.5199723  -0.50368898 -0.49662209 -0.43643029 -0.19703128]]
- **The value of the last pixel**: 1.1028259918382395
