import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the TimeSeriesObject instances from the saved file
with open('1000_data_objects.pkl', 'rb') as file:
    normalized_time_series_objects = pickle.load(file)

data = np.array([
    np.hstack((ts_obj.getFirstFifteenMinutesXp(),
               ts_obj.getFirstFifteenMinutesGold(),
               ts_obj.getFirstFifteenMinutesCs())) for ts_obj in normalized_time_series_objects])
labels = np.array([ts_obj.classification for ts_obj in normalized_time_series_objects])

# Stratified train-test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(data, labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Initialize and train the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=88)
random_forest.fit(X_train, y_train)

# Make predictions on the test set and calculate accuracy
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy with Random Forest Classifier: {accuracy_rf}")
print()

# Calculate confusion matrix for Random Forest Classifier
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Random Forest Classifier):")
print(conf_matrix_rf)

print()

# Generate and print classification report for Random Forest Classifier
class_report_rf = classification_report(y_test, y_pred_rf)
print("Classification Report (Random Forest Classifier):")
print(class_report_rf)
