import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the TimeSeriesObject instances from the saved file
with open('1500_data_objects.pkl', 'rb') as file:
    normalized_time_series_objects = pickle.load(file)

data = np.array([
    np.hstack((ts_obj.getFirstFifteenMinutesXp(),
               ts_obj.getFirstFifteenMinutesGold(),
               ts_obj.getFirstFifteenMinutesCs())) for ts_obj in normalized_time_series_objects])
labels = np.array([ts_obj.classification for ts_obj in normalized_time_series_objects])

# Stratified train-test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=33)
for train_index, test_index in sss.split(data, labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Initialize and train the kNN classifier with k=30
knn_classifier = KNeighborsClassifier(n_neighbors=38, weights="distance")
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set and calculate accuracy
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with 7 neighbors: {accuracy}")
print()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print()

# Generate and print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
