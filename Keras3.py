import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# Original data set retrieved from here:
# https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line


################################################
######Loading and Plotting the Data#############
################################################


# data = pd.read_csv("Opendata Bahn/international-airline-passengers.csv",
#                      usecols = [1],
#                      engine = "python",
#                      skipfooter = 3)
# data


data = pd.read_csv("Opendata Bahn/stationsListFfmTotal.csv",
                   usecols=[1],
                   engine="python",
                   skipfooter=3)
data.head()


# Create a time series plot.
plt.figure(figsize=(15, 5))
plt.plot(data, label="Airline Passengers")
plt.xlabel("Days")
plt.ylabel("1000 International Airline Passengers")
plt.title("Monthly Total Airline Passengers 1949 - 1960")
plt.legend()
# plt.show()


################################################
################################################
######     Building LST Model      #############
################################################
################################################


# Let's load the required libs.
# We'll be using the Tensorflow backend (default).
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


################################################
######       Data Preparation      #############
################################################

# Get the raw data values from the pandas data frame.

data_raw = data.values.astype("float32")
data_raw
# We apply the MinMax scaler from sklearn
# to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[0:5]

################################################
####   Split into Test/Training Data    ########
################################################

# As usual, the data gets split into training and test data so we can later assess how well the final model performs.
# Again, this could be much improved, e.g. using CV and more sophisticated steps to select the "best" model.

# Using 60% of data for training, 40% for validation.
TRAIN_SIZE = 0.60

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " +
      str((len(train), len(test))))


################################################
####  Get Data into Shape for Keras     ########
################################################

# FIXME: This helper function should be rewritten using numpy's shift function. See below.
def create_dataset(dataset, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))


# Create test and training sets for one-step-ahead regression.
window_size = 2
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)


################################################
####  Build LSTM Model on train data    ########
################################################

# The LSTM architecture here consists of:

# One input layer.
# One LSTM layer of 4 blocks.
# One Dense layer to produce a single output.
# Use MSE as loss function.

# Many different architectures could be considered. But this is just a quick test, so we'll keep things nice and simple.

def fit_model(train_X, train_Y, window_size=1):
    model = Sequential()

    model.add(LSTM(5, input_shape=(1, window_size), return_sequences=True))
    # inout shape = (time_steps, vector_size)
    model.add(LSTM(10))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",
                  optimizer="adam")
    model.fit(train_X,
              train_Y,
              epochs=120,
              batch_size=1,
              verbose=2)

    return (model)


# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)


################################################
################################################
######         Result              #############
################################################
################################################


################################################
####  Prediction and model evaluation   ########
################################################

def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)


rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)


################################################
##  Plot original data, prediction, forecast ###
################################################

# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(
    train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) +
                  1:len(dataset) - 1, :] = test_predict

# Create the plot.
plt.figure(figsize=(15, 5))
plt.plot(scaler.inverse_transform(dataset), label="True value")
plt.plot(train_predict_plot, label="Training set prediction")
plt.plot(test_predict_plot, label="Test set prediction")
plt.xlabel("Months")
plt.ylabel("1000 International Airline Passengers")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()
