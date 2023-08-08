from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()


# Load the dataset
data = pd.read_csv("Housing.csv")

# Preprocess features with text format using LabelEncoder
le = LabelEncoder()
TextColumns = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]
for col in TextColumns:
    data[col] = le.fit_transform(data[col])

# Split the data
X = data.drop('price', axis=1)
y = data['price']
std = StandardScaler()

X = scaler.fit_transform(X)
pca = PCA(n_components=X.shape[1])
X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)



clf = LinearRegression()
clf.fit(X_train, y_train)



# Make predictions using the trained model
predictions = clf.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)

# Calculate the accuracy (you might want to use a different metric for regression)
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    to_predict = scaler.transform(to_predict)
    to_predict = pca.transform(to_predict)
    result = clf.predict(to_predict)
    return result[0]

def GetMeanHousePrice(data = data['price']):
    return data.mean()
