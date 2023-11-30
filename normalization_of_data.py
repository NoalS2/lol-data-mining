import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the TimeSeriesObject instances from the saved file
with open('1500_data_objects.pkl', 'rb') as file:
    time_series_objects = pickle.load(file)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Perform normalization on each TimeSeriesObject's gold
for ts_object in time_series_objects:
    # Reshape and normalize the gold attribute
    ts_object.gold = scaler.fit_transform(ts_object.gold.reshape(-1, 1)).flatten()

    # Reshape and normalize the xp attribute
    ts_object.xp = scaler.fit_transform(ts_object.xp.reshape(-1, 1)).flatten()

# Save the normalized TimeSeriesObject instances
with open('1500_data_normalized_objects.pkl', 'wb') as file:
    pickle.dump(time_series_objects, file)
