import pandas as pd
import numpy as np

#Input dataset
data = pd.read_csv('knnData.csv')

#Extract training data and test data
X_train = data[['trainPoints_x1', 'trainPoints_x2']].values
Y_train = data['trainLabel'].values

X_test = data[['testPoints_x1','testPoints_x2']].values
Y_test = data['testLabel'].values

#Calculate the distance, firstly, define distance functions 
def L1_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))    #Manhattan distance
def L2_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))    #Euclidean distance
def Linf_distance(x1, x2):
    return np.max(np.abs(x1 - x2))    #Chebyshev distance

#Secondly, distance-weighted KNN implementation
def weighted_knn(X_train, Y_train, X_test, Y_test, k, distance_function):
    correct_predictions = 0

    #iterate through each test point
    for i, test_point in enumerate(X_test):
        distances = []
        for j, train_point in enumerate(X_train):
            dist = distance_function(test_point, train_point)
            distances.append((dist,Y_train[j]))
            
        
        #Sort distances and select the k nearest neighbors
        distances.sort(key = lambda x : x[0])
        k_neighbors = distances[:k]


        label_weights = {}
        for dist, label in k_neighbors:
            weight = 1 / ((dist + 1e-5) ** 2)
            label_weights[label] = label_weights.get(label, 0) + weight

        predicted_label = max(label_weights, key= label_weights.get)
        
        if predicted_label == Y_test[i]:
            correct_predictions += 1
    #Calculate accuracy
    accuracy = correct_predictions / len(Y_test)
    return accuracy

#Evaluate the model with different distance metrics
k = 3
distance_metrics = {
    "L1": L1_distance,
    "L2": L2_distance,
    "Linf": Linf_distance
}
#Run KNN for each distance metrics and calculate accuracy
for metric_name, metric_func in distance_metrics.items():
    accuracy = weighted_knn(X_train, Y_train, X_test, Y_test, k, metric_func)
    print(f"Accuracy with {metric_name}\nDistance:{accuracy:.2%}")