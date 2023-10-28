import numpy as np
import pandas as pd

# loading the dataset
train = pd.read_csv('train_mnist.csv',skiprows=1)
test = pd.read_csv('test_mnist.csv',skiprows=1)

X_Train = train.iloc[:, 1:].values
Y_Train = train.iloc[:, 0].values
x_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True) * (1 - exp_x / np.sum(exp_x, axis=1, keepdims=True))

def weights(x, y):
    weight = np.random.uniform(-1., 1., size=(x, y)) / np.sqrt(x * y)
    return weight.astype(np.float32)

def feedforward(x, w1, w2):
    input_layer = x.dot(w1)
    input_activation = sigmoid(input_layer)

    hidden_layer = input_activation.dot(w2)
    output = softmax(hidden_layer)

    return input_activation, output

def backpropagation(x, y, input_activation, output, w1, w2, learn_rate):
    target = np.zeros((len(y), 10), np.float32)
    #print(target.shape)
    for i in range(len(y)):
        target[i, y[i]] = 1.0

    output_error = (output - target) / len(x)
    #    print(output_error.shape,input_activation.shape,x.shape)
    hidden_error = output_error.dot(w2.T) * sigmoid_derivative(input_activation)

    w2_error = input_activation.T.dot(output_error)
    w1_error = x.T.dot(hidden_error)

    w2 -= learn_rate * w2_error
    w1 -= learn_rate * w1_error

w1 = weights(784, 128)
w2 = weights(128, 10)

epochs = 10000
learn_rate = 0.001
sample_size = 256

ac = []

for i in range(epochs):
    random_sample = np.random.randint(0, X_Train.shape[0], size=(sample_size))
    x = X_Train[random_sample].reshape((-1, 28 * 28))
    y = Y_Train[random_sample]


    inputs, out = feedforward(x, w1, w2)
    backpropagation(x, y, inputs, out, w1, w2, learn_rate)

    category = np.argmax(out, axis=1)
    accuracy = (category == y).mean()

    if (i % 500 == 0):
        print("epoch", i, "train accuracy:", accuracy)