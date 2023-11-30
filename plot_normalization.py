import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the TimeSeriesObject instances from the saved file
with open('500_data_objects.pkl', 'rb') as file:
    time_series_objects = pickle.load(file)

with open('500_data_normalized_objects.pkl', 'rb') as file:
    normalized_time_series_objects = pickle.load(file)

original_data = time_series_objects[0].gold.copy()

normalized_data = normalized_time_series_objects[0].gold.copy()

# Plotting original and normalized data
plt.figure(figsize=(10, 5))

# Plot original data
plt.subplot(1, 2, 1)
plt.plot(original_data, label='Original Data')
plt.title('Original Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plot normalized data
plt.subplot(1, 2, 2)
plt.plot(normalized_data, label='Normalized Data')
plt.title('Normalized Data')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend()

plt.tight_layout()
plt.show()
