from sys import argv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD, Adam
from keras.models import load_model
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pandas import DataFrame


# configure graphing visuals/size
sns.set_style("darkgrid")
plt.figure(figsize=(12, 5))


def read_stocks(file_path, column):
    """ 
    Creates list of values from stock csv - possible columns:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    values = []
    with open(file_path) as file:
        reader = csv.DictReader(file)
        values = [line[column] for line in reader]
    return values


def build_data_set(values, window):
    """
    Breaks the values into the data and labels with the time window
    (window) days of data, next day as the label
    """
    data, labels = [], []
    for i in range(len(values) - window):
        data.append(values[i:i + window])
        labels.append([values[i + window]])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def normalize(scaler, values):
    """
    Normalize values in a list to the range (0, 1)
    Have to reshape for how normalization is done
    """
    if scaler is None:
        scaler = StandardScaler()
    count = len(values)
    values = np.array(values)
    values = scaler.fit_transform(values.reshape(-1, 1))
    values = np.array(values).reshape(count,)
    return scaler, values


def denormalize(scaler, values):
    """
    Denormalizes values back to original values, using
    given Scaler (same used to normalize)
    """
    if type(values) != list:
        values = [values]
    return scaler.inverse_transform(values)


def create_model(input_shape):
    """
    Creates LSTM model with input_shape, the
    Mean Squared Log Error loss and Adam optimizer
    """
    model = Sequential([
        LSTM(16, input_shape=input_shape, return_sequences=True),
        LSTM(16, return_sequences=True),
        LSTM(16, return_sequences=False),
        Dense(1)
    ])
    model.compile(loss='msle', optimizer='adam')
    return model


def train_model(model, data, labels, epochs, batch_size):
    """
    Trains model with given data, labels, etc.
    """
    data_size = data.shape[0]   # rows: amount of data
    time_steps = data.shape[1]  # cols: number of time steps
    features = 1                # each time step has one piece of data (stock value)
    data = data.reshape(data_size, time_steps, features)
    model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return model


def line_graph(data):
    """
    Creates a Pandas DataFrame out of the data and plots
    each key/points as a line graph (solid lines)
    """
    df = DataFrame(data=data)
    sns.lineplot(data=df, dashes=False)
    plt.show()


def driver_train():
    """
    Common Testing: Create and train a model with the high
    stock values, and save the model to an h5 file after
    """
    # get the stock values, normalize, and train model
    stocks = read_stocks('stocks.csv', 'High')
    scaler, scaled_values = normalize(None, stocks)
    window = 30     # 30 day time frame
    data, labels = build_data_set(scaled_values, window)
    model = create_model((window, 1))
    train_model(model, data, labels, 30, 30)
    model.save('model.h5')  # save model for easy loading


def driver_graph():
    """
    Common Testing: Loads in saved model and for every 30 days,
    it predicts the next day's high stock value. Graphs the 
    predicted vs actual values
    """
    # use existing model to test/graph data
    stocks = read_stocks('stocks.csv', 'High')
    scaler, scaled_values = normalize(None, stocks)
    window = 30
    model = load_model('model.h5')
    data = {
        'Actual': [],
        'Prediction': []
    }
    # run through scaled values, with a window size
    for i in range(len(scaled_values) - window):
        values = np.array([scaled_values[i:i + window]])
        data_size, time_steps = values.shape[0], values.shape[1]
        reshaped = values.reshape(data_size, time_steps, 1)
        result = model.predict(reshaped)[0]     # predict next day's stock value
        # denormalize for real/predicted answer
        answer = denormalize(scaler, scaled_values[i + window])[0]#[0]
        prediction = denormalize(scaler, result)[0][0]
        data['Actual'].append(answer)
        data['Prediction'].append(prediction)
    # graph the actual vs predicted values
    line_graph(data)


def tomorrow():
    """
    Common Testing: Loads in saved model, and uses the latest
    30 day period to predict the next day's high stock value
    """
    stocks = read_stocks('stocks.csv', 'High')
    scaler, scaled_values = normalize(None, stocks)
    window = 30
    # get the latest 30 days (window size)
    values = np.array([scaled_values[-window:]])
    data_size, time_steps = values.shape[0], values.shape[1]
    reshaped = values.reshape(data_size, time_steps, 1)
    model = load_model('model.h5')
    # predict new value and denormalize
    result = model.predict(reshaped)[0]
    prediction = denormalize(scaler, result)[0][0]
    print(prediction)


if __name__ == "__main__":
    # maps command line arguments to functions and runs 
    # the corresponding ones (ex. 'train graph' would 
    # train the model and then graph in that order)
    args = argv[1:]
    commands = {
        'train': driver_train,
        'graph': driver_graph,
        'tomorrow': tomorrow
    }
    args = args if len(args) > 0 else ['train', 'graph']
    for arg in args:
        commands[arg]()

