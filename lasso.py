import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

def pretreat(X, method, para1=None, para2=None):
    if method == 1:  # Autoscaling
        if para1 is None and para2 is None:
            para1 = np.mean(X, axis=0)
            para2 = np.std(X, axis=0)
    elif method == 2:  # Centering
        if para1 is None and para2 is None:
            para1 = np.mean(X, axis=0)
            para2 = np.ones(X.shape[1])
    elif method == 3:  # Unilength
        if para1 is None and para2 is None:
            para1 = np.mean(X, axis=0)
            para2 = np.array([np.linalg.norm(X[:, j] - para1[j]) for j in range(X.shape[1])])
    elif method == 4:  # Min-Max scaling
        if para1 is None and para2 is None:
            para1 = np.min(X, axis=0)
            para2 = np.max(X, axis=0) - para1
    elif method == 5:  # Pareto scaling
        if para1 is None and para2 is None:
            para1 = np.mean(X, axis=0)
            para2 = np.sqrt(np.std(X, axis=0))
    else:
        print('Wrong data pretreat method!')
        return

    X = (X - para1) / para2
    return X, para1, para2

def plot_results(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.show()

# Load your dataset
df = pd.read_csv(r'sonar.csv')
X = df.iloc[:, :-1].values  # Convert DataFrame to NumPy array
y = df.iloc[:, -1].values

# Convert the categorical target variable into numerical form using label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Choose your method for data pretreatment
print("Choose data pretreatment method:")
print("1. Autoscaling")
print("2. Centering")
print("3. Unilength")
print("4. Min-Max Scaling")
print("5. Pareto Scaling")
method = int(input("Enter the method number (1-5): "))

# Pretreat the data
X, para1, para2 = pretreat(X, method)

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train to a NumPy array
y_train = np.array(y_train, dtype=float).reshape(-1, 1)

# Lasso Regression

m, n = X_train.shape
theta = np.zeros(n) + 0.0
lambda_param = 0.008 # Adjust this parameter as needed
num_iters = 100  # Adjust this parameter as needed

for _ in range(num_iters):
    for j in range(n):
        tmp_theta = theta.copy()
        tmp_theta[j] = 0.0
        r_j = y_train.ravel() - X_train.dot(tmp_theta)
        arg1 = np.dot(X_train[:, j], r_j)
        arg2 = lambda_param * m

        if np.any(arg1 < -arg2):
            theta[j] = (arg1 + arg2) / (X_train[:, j]**2).sum()
        elif np.any(arg1 > arg2):
            theta[j] = (arg1 - arg2) / (X_train[:, j]**2).sum()
        else:
            theta[j] = 0.0

theta_best = theta

# Predicting the Test set results
y_pred = X_test.dot(theta_best)

# Plotting the results
# plot_results(y_test, y_pred)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Predicting the Training set results
y_train_pred = X_train.dot(theta_best)

print(f"Theta Best: {theta_best}")

# Create a mask for coefficients that are not equal to zero
mask = theta_best != 0

# Use the mask to filter out zero coefficients
theta_best_non_zero = theta_best[mask]

print(f"Theta Best NON ZERO: {theta_best_non_zero}")

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors as needed

# Use only the features corresponding to the non-zero coefficients for training
X_train_non_zero = X_train[:, mask]
X_test_non_zero = X_test[:, mask]

# Fit the model to the training data
knn.fit(X_train_non_zero, y_train.ravel())

y_pred_knn = knn.predict(X_test_non_zero)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_knn)

print(f"KNN Accuracy: {accuracy}")
