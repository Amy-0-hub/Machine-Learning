import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score



# data exploration
trainData =pd.read_csv('trainInput.csv', header=None)
trainDataLabels=pd.read_csv('trainOutput.csv', header=None)
testData = pd.read_csv('testInput.csv', header=None)
testDataLabels =pd.read_csv('testOutput.csv', header=None)

#print out the shapes of the imports
print(trainData.shape)
print(trainDataLabels.shape)
print(testData.shape)
print(testDataLabels.shape)

#check the values of labels of train data
trainDataLabels.head()

#Create a matrix A where each row corresponds to a training images of that digit.
DigitArrays = {}
for i in range(10): 
    DigitArrays[i] = []
for i in range(len(trainDataLabels)):  
    DigitArrays[trainDataLabels[i]].append(trainData[:,i])
for i in DigitArrays: 
    DigitArrays[i] = np.array(DigitArrays[i])
DigitArrays[0]

# singular value decomposition for each A and express those as linear combination of the first k=20 singular images of each digit.
svd_results = {} 
for digit, images in DigitArrays.items(): 
    U, S, Vt = np.linalg.svd(images, full_matrices=False) 
    svd_results[digit] = Vt[:20]  # Select first 20 singular vectors
svd_results[0]

#Compute the least square distance and classify test images
predict=[] 
for test in testData.T: 
    residuals = {} 
    for digit, vector in svd_results.items(): 
        proj = vector.T @ (vector @ test) 
        residuals[digit]=np.linalg.norm(test - proj) 
    predicted = min(residuals, key=residuals.get) #compare residuals against each other to find the best prediction.
    predict.append(predicted) #Add prediction to the array
predict[0:20] #Display predictions Vs actual results for transarency

testDataLabels[0:20]

	
#Calculate the overall correct classification rate, as well as correct classification rate for each digit in a confusion matrix.
accuracy = accuracy_score(testDataLabels, predict) #use sklearn's accuracy score to find out how correct our prediction was.
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(testDataLabels, predict) #use sklearn's confusion matrix to create a confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

each_accuracy={} 
for digit in range(0,10): 
    correct_rate = conf_matrix[digit, digit] 
    total = np.sum(conf_matrix[digit, :]) 
    each_accuracy[digit] = correct_rate / total 
for key, value in each_accuracy.items(): 
    print(f'({key}) digit accuracy is {value * 100:.2f}%')