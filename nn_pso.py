# Import modules
import numpy as np
#import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
import pandas as pd
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

def load_iris_dataset():
    data = pd.read_csv("iris.data", header=None)
    output_str = data.iloc[:, 4]
    #data = np.array(data.iloc[:, 0:4])
    output = []
    for i in output_str:
        if i == "Iris-setosa":
            output.append(0)
        elif i == "Iris-versicolor":
            output.append(1)
        else:
            output.append(2)
    output = np.array(output)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(37):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(37, 50):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    for i in range(50, 87):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(87, 100):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    for i in range(100, 138):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(138, 150):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 4
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes

def load_wine_dataset():
    input_data = pd.read_csv("wine.data", header=None)
    data = (input_data.iloc[:, 1:14])
    data_output = (input_data.iloc[:, 0])
    output = []
    for i in data_output:
        output.append(i - 1)
    output = np.array(output)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(44):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    for i in range(44, 59):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    print(y_test[-1])
    for i in range(59, 112):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    print(y_train[43])
    for i in range(112, 130):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    for i in range(130, 166):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    for i in range(166, 178):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 13
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes

def load_cancer_dataset():
    input_data = pd.read_csv("breast-cancer-wisconsin.data", header=None, skiprows=[23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315, 321, 411, 617])
    benign_data = []
    malignant_data = []
    for i in range(input_data.shape[0]):
        if input_data.iloc[i, 10] == 2:
            benign_data.append(input_data.iloc[i, 0 : -1])
        else:
            malignant_data.append(input_data.iloc[i, 0 : -1])
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(int(0.75 * len(benign_data))):
        x_train.append(benign_data[i])
        y_train.append(1)
    for i in range(int(0.75 * len(malignant_data))):
        x_train.append(malignant_data[i])
        y_train.append(0)
    for i in range(int(0.75 * len(benign_data)) + 1 , len(benign_data)):
        x_test.append(benign_data[i])
        y_test.append(1)
    for i in range(int(0.75 * len(malignant_data)) + 1 , len(malignant_data)):
        x_test.append(malignant_data[i])
        y_test.append(0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 10
    n_classes = 2
    return x_train, y_train, x_test, y_test, n_inputs, n_classes   

x_train = None 
y_train = None 
x_test = None
y_test = None
n_inputs = None
n_classes = None
n_hidden = None
#x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_iris_dataset()
#x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_wine_dataset()

# Forward propagation
def forward_prop(params):
    global n_inputs
    global n_classes
    global n_hidden
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """
    # Neural network architecture
    n_inputs = n_inputs
    n_hidden = n_hidden
    n_classes = n_classes

    # Roll-back the weights and biases
    #W1 = params[0:20].reshape((n_inputs,n_hidden))
    W1 = params[0:n_inputs * n_hidden].reshape((n_inputs,n_hidden))
    #b1 = params[20:25].reshape((n_hidden,))
    b1 = params[n_inputs * n_hidden:(n_inputs * n_hidden) + n_hidden].reshape((n_hidden,))
    #W2 = params[25:40].reshape((n_hidden,n_classes))
    W2 = params[(n_inputs * n_hidden) + n_hidden:((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes].reshape((n_hidden,n_classes))
    #b2 = params[40:43].reshape((n_classes,))
    b2 = params[(((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes):(((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes) + n_classes].reshape((n_classes,))
    # Perform forward propagation
    z1 = x_train.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1.astype(np.float64))     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = len(x_train) # Number of samples
    corect_logprobs = -np.log(probs[range(N), y_train])
    loss = np.sum(corect_logprobs) / N
    #print(loss)
    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)


def predict(X, pos):
    global n_inputs
    global n_classes
    global n_hidden
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = n_inputs
    n_hidden = n_hidden
    n_classes = n_classes

    # Roll-back the weights and biases
    #W1 = params[0:20].reshape((n_inputs,n_hidden))
    W1 = pos[0:n_inputs * n_hidden].reshape((n_inputs,n_hidden))
    #b1 = params[20:25].reshape((n_hidden,))
    b1 = pos[n_inputs * n_hidden:(n_inputs * n_hidden) + n_hidden].reshape((n_hidden,))
    #W2 = params[25:40].reshape((n_hidden,n_classes))
    W2 = pos[(n_inputs * n_hidden) + n_hidden:((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes].reshape((n_hidden,n_classes))
    #b2 = params[40:43].reshape((n_classes,))
    b2 = pos[(((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes):(((n_inputs * n_hidden) + n_hidden) + n_hidden * n_classes) + n_classes].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 =  np.tanh(z1.astype(np.float64))     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred

def execute_pso(index):
    global x_train
    global y_train
    global x_test
    global y_test
    global n_inputs
    global n_classes
    global n_hidden
    if index == 0:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_iris_dataset()
    elif index == 1:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_wine_dataset()
    else:
        x_train, y_train, x_test, y_test, n_inputs, n_classes = load_cancer_dataset()
    # Initialize swarm
    if index == 2:
        options = {'c1': 0.8, 'c2': 0.2, 'w':0.9, 'k':3 , 'p': 1}
        print("ENTREI !!!")
        n_hidden = 6
    else:
        options = {'c1': 0.7, 'c2': 0.2, 'w':0.95, 'k':3 , 'p': 1}
        n_hidden = 5
    
    # Call instance of PSO
    dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes
    start_time = time.time()
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)
    iter_n = []
    iter_n.append(5000)
    if index == 0:
        iter_n[0] = 1000

    # Perform optimization
    cost, pos = optimizer.optimize(f,  iters=iter_n[0], verbose=3)
    time_seconds = (time.time() - start_time)
    
    y_pred = predict(x_test, pos)
    acc_test = (accuracy_score(y_test, y_pred))
    f1_test = (f1_score(y_test, y_pred, average='macro'))
    prec_test = (precision_score(y_test, y_pred, average='macro'))
    rec_test = (recall_score(y_test, y_pred, average='macro'))
    y_pred = predict(x_train, pos)
    acc_train = (accuracy_score(y_train, y_pred))
    f1_train = (f1_score(y_train, y_pred, average='macro'))
    prec_train = (precision_score(y_train, y_pred, average='macro'))
    rec_train = (recall_score(y_train, y_pred, average='macro'))
    return time_seconds , optimizer.cost_history, acc_test, f1_test, prec_test, rec_test, acc_train, f1_train, prec_train, rec_train




