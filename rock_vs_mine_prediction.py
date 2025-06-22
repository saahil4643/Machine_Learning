# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
sonar_data = pd.read_csv('/content/sample_data/sonar data.csv')

# Display the first 5 rows of the dataset
sonar_data.head()

# Show the shape (number of rows and columns) of the dataset
sonar_data.shape

# Generate descriptive statistics for each feature
sonar_data.describe()

# Print all column names
print(sonar_data.columns)

# Show count of each class in the target column 'R'
print(sonar_data['R'].value_counts())

# Show the mean of each feature grouped by the target label 'R'
print(sonar_data.groupby('R').mean())

# Split the data into features (X) and target (Y)
X = sonar_data.drop(columns='R', axis=1)
Y = sonar_data['R']

# Print features and labels
print(X)
print(Y)

# Split the dataset into training and test sets (90% train, 10% test), stratified by label
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model with training data
model.fit(X_train, Y_train)

# Predict on training data
X_train_prediction = model.predict(X_train)

# Calculate accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

# ------------------ Prediction for new input data ------------------

# Input data (60 features)
input_data = (0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
              0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
              0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
              0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
              0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
              0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032)

# Convert input data to numpy array
input_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
reshaped_array = input_numpy_array.reshape(1, -1)

# Make prediction
prediction = model.predict(reshaped_array)
print('Predicted class:', prediction[0])
