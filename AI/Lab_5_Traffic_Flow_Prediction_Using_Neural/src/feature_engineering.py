import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# STEP 2: Feature Engineering
def create_time_features(data):
    # Extract additional time-based features
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    return data

def create_lag_features(data, lag=1):
    for i in range(1, lag + 1):
        data[f'traffic_flow_lag_{i}'] = data['traffic_flow'].shift(i)
    data.dropna(inplace=True)
    return data

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Load preprocessed data
    file_path = 'D:/Data/Lab_5/synthetic_traffic_data - synthetic_traffic_data.csv'
    data = pd.read_csv(file_path)
    
    # Preprocess data
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.sort_values(by='timestamp', inplace=True)
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data.ffill(inplace=True)
    
    # Feature engineering
    data = create_time_features(data)
    data = create_lag_features(data, lag=3)
    
    # Split data
    X = data.drop(['traffic_flow', 'timestamp'], axis=1)
    y = data['traffic_flow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    print("Feature Engineering Completed")