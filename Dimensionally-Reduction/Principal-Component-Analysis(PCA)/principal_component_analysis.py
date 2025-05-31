import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px

#image compression
path='/Users/yaqin/Downloads/landset_images'
X=[]
for imgFile in os.listdir(path):
    #read all images in the folder
    imgArray=np.array(Image.open(os.path.join(path,imgFile)))
    #stack all images together
    X.append(imgArray)
    fig1=px.imshow(imgArray, binary_string=True)
    fig1.update_layout(title=imgFile)
    fig1.show()

#set up a pipeline that scales the values linearly between 0 and 1 and applies PCA such that 5 images from different wavelengths are combined to 1 channel data. 
images = np.array(X)
print(images.shape)

my_pca_pipeline = Pipeline(
    steps=[
        ('scale', MinMaxScaler()), #scale between 0 and 1
        ('pca', PCA(n_components=1))
    ]
)

# We must stack up all pixels so they're 1D array. i.e. each image's dimension is 1 by height * width * depth
# In this case depth=1
images_train=images.reshape(images.shape[0], images.shape[1] * images.shape[2])
images_train.shape
images_train = images_train.T


my_pca_pipeline.fit(images_train)

#report the number of features and the number of samples
print('Number of features:' , my_pca_pipeline[-1].n_features_in_)
print('Number of samples:', my_pca_pipeline[-1].n_samples_)
print('PCA direction:', my_pca_pipeline[-1].components_)
print('ratio:', my_pca_pipeline[-1].explained_variance_ratio_)

#compute the value of the last pixel
images_transform = my_pca_pipeline.transform(images_train)
images_transform.shape
last_pixel = images_transform[-1,-1]
last_pixel

fig3 = px.imshow(images_transform.reshape(7651, 7551), binary_string=True)
fig3.show()