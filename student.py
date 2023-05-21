import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


# scroll to the bottom to start coding your solution


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # write your code here
def scale(X_train, X_test):
    return X_train/np.max(X_train), X_test/np.max(X_test)


def xavier(n_in, n_out):
    value = np.sqrt(6)/np.sqrt(n_in + n_out)
    return np.random.uniform(-value,value,(n_in, n_out))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y, pred):
    return np.mean((y - pred)**2)


def mse_der(y, pred):
    return 2 * np.subtract(y, pred)


def result(lst):
    return [1 if x == np.max(lst) else 0 for x in lst]


def train(model, X, y, alpha, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)


def accuracy(model, X, y):
    y_pred = np.argmax(model.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)
        self.output = 0

    def forward(self, X):
        X = sigmoid(np.dot(X, self.weights) + self.biases)
        self.output = X
        return X

    def backprop(self, X, y, alpha):
        error = mse_der(self.forward(X), y) * sigmoid_der(np.dot(X, self.weights) + self.biases)
        self.weights -= alpha * np.dot(X.T, error) / X.shape[0]
        self.biases -= alpha * np.mean(error)


class TwoLayerNeural:
    def __init__(self, n_features, n_classes, hidden_layer=64):
        self.weights_first = xavier(n_features, hidden_layer)
        self.biases_first = xavier(1, hidden_layer)
        self.weights_second = xavier(hidden_layer, n_classes)
        self.biases_second = xavier(1, n_classes)
        self.first_step = 0
        self.sec_step = 0

    def forward(self, X):
        self.first_step = sigmoid(np.dot(X, self.weights_first) + self.biases_first)
        self.sec_step = sigmoid(np.dot(self.first_step, self.weights_second) + self.biases_second)
        return self.sec_step

    def backprop(self, X, y, alpha):
        error = mse_der(self.forward(X), y) * sigmoid_der(np.dot(self.first_step, self.weights_second) + self.biases_second)
        self.weights_second -= alpha * np.dot(self.first_step.T, error) / X.shape[0]
        self.biases_second -= alpha * np.mean(error)
        error2 = np.dot(error, self.weights_second.T) * sigmoid_der(np.dot(X, self.weights_first))
        self.weights_first -= alpha * np.dot(X.T, error2) / X.shape[0]
        self.biases_first -= alpha * np.mean(error2)

'''
r1 = accuracy(model, X_test, y_test).flatten().tolist()
r2 = []
for _ in range(20):
    train(model, X_train, y_train, 0.5)
    r2.append(accuracy(model, X_test, y_test))

print(r1, r2)

'''
X_train, X_test = scale(X_train, X_test)


model1 = TwoLayerNeural(X_train.shape[1], 10)
model1.forward(X_train[:2])

r1 = accuracy(model1, X_test, y_test).flatten().tolist()
r2 = []
for _ in range(20):
    train(model1, X_train, y_train, 0.5)

    r2.append(accuracy(model1, X_test, y_test))

print(r2)



