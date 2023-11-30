import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the TimeSeriesObject instances from the saved file
with open('1500_data_objects.pkl', 'rb') as file:
    normalized_time_series_objects = pickle.load(file)

data = np.array([
    np.hstack((ts_obj.getFirstFifteenMinutesXp(),
               ts_obj.getFirstFifteenMinutesGold(),
               ts_obj.getFirstFifteenMinutesCs())) for ts_obj in normalized_time_series_objects])

labels = np.array([ts_obj.classification for ts_obj in normalized_time_series_objects])

# Initialize a range of k values to test
k_values = list(range(1, 76))  # Test k from 1 to 50

train_accuracies = []
test_accuracies = []

for k in k_values:
    # Initialize lists to store accuracies for each fold
    fold_train_accuracies = []
    fold_test_accuracies = []

    # Stratified train-test split for each k value
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    for train_index, test_index in sss.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Initialize and train the kNN classifier for the current k
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

        # Make predictions on the train and test sets for the current k
        y_train_pred = knn_classifier.predict(X_train)
        y_test_pred = knn_classifier.predict(X_test)

        # Calculate accuracy for train and test sets for the current k and store it
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        fold_train_accuracies.append(train_accuracy)
        fold_test_accuracies.append(test_accuracy)

    # Calculate mean accuracies for the current k and store them
    mean_train_accuracy = np.mean(fold_train_accuracies)
    mean_test_accuracy = np.mean(fold_test_accuracies)

    train_accuracies.append(mean_train_accuracy)
    test_accuracies.append(mean_test_accuracy)

# Plotting k vs. Accuracy for both training and testing sets
plt.figure(figsize=(16, 6))
plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(k_values, test_accuracies, marker='o', label='Test Accuracy')
plt.title('k vs. Accuracy for kNN (Train and Test)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
