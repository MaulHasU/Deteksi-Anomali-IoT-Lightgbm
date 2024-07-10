import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
file_path = 'data/data_trainv4.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Normalize the data (excluding the 'devices' column)
scaler = MinMaxScaler()
data_scaled = data.copy()
data_scaled.iloc[:, 2:] = scaler.fit_transform(data_scaled.iloc[:, 2:])

# Prepare the data for training
X = data_scaled.drop(columns=['devices', 'flag'])
y = data_scaled['flag']

# Balance the dataset if needed (optional, depends on the imbalance)
# Here we use oversampling of the minority class as an example
from sklearn.utils import resample

data_majority = data_scaled[data_scaled.flag == 0]
data_minority = data_scaled[data_scaled.flag == 1]

data_minority_upsampled = resample(data_minority,
                                   replace=True,     # sample with replacement
                                   n_samples=len(data_majority),    # to match majority class
                                   random_state=42) # reproducible results

data_upsampled = pd.concat([data_majority, data_minority_upsampled])

X_upsampled = data_upsampled.drop(columns=['devices', 'flag'])
y_upsampled = data_upsampled['flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Train the LightGBM classifier with GridSearchCV for hyperparameter tuning
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500]
}

lgbm = LGBMClassifier(random_state=42)
clf = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

# Save the best model
joblib.dump(clf.best_estimator_, 'models/lgbm_model.pkl')

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
