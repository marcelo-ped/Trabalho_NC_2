import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_score, recall_score
import time
"""
def load_iris_dataset():
    data = pd.read_csv("iris.data", header=None)
    output_str = data.iloc[:, 4]
    data = numpy.array(data.iloc[:, 0:4])
    print(data[0])
    print((data.shape))
    output = []
    for i in output_str:
        if i == "Iris-setosa":
            output.append(0)
        elif i == "Iris-versicolor":
            output.append(1)
        else:
            output.append(2)
    output = numpy.array(output)
    return data, output
"""

def load_iris_dataset():
    print("ENREI IRIS DATA")
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
    output = numpy.array(output)
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
    x_train = numpy.asarray(x_train).astype('float32')
    x_test = numpy.asarray(x_test).astype('float32')
    y_train = numpy.asarray(y_train).astype('float32')
    y_test = numpy.asarray(y_test).astype('float32')
    perm = numpy.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = numpy.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 4
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes

def load_wine_dataset():
    print("ENTREI WINE DATA")
    input_data = pd.read_csv("wine.data", header=None)
    data = (input_data.iloc[:, 1:14])
    data_output = (input_data.iloc[:, 0])
    output = []
    for i in data_output:
        output.append(i - 1)
    output = numpy.array(output)
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
    x_train = numpy.asarray(x_train).astype('float32')
    x_test = numpy.asarray(x_test).astype('float32')
    y_train = numpy.asarray(y_train).astype('float32')
    y_test = numpy.asarray(y_test).astype('float32')
    perm = numpy.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = numpy.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 13
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes

def load_cancer_dataset():
    print("ENTREI CANCER DATA")
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
    x_train = numpy.asarray(x_train).astype('float32')
    x_test = numpy.asarray(x_test).astype('float32')
    y_train = numpy.asarray(y_train).astype('float32')
    y_test = numpy.asarray(y_test).astype('float32')
    perm = numpy.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = numpy.random.permutation(len(x_test))
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
model = None
def fitness_func(solution, sol_idx):
    global x_train, y_train, keras_ga, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    predictions = model.predict(x_train)

    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / (cce(y_train, predictions).numpy() + 0.00000001)

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Build the keras model using the functional API.







"""
x, y  = load_iris_dataset()
y = tensorflow.keras.utils.to_categorical(y)
#x = normalize_data(x)
perm = numpy.random.permutation(len(x))
x_train = x[perm[:int(0.75 * len(perm))]]
y_train = y[perm[:int(0.75 * len(perm))]]
x_test = x[perm[int(0.75 * len(perm) + 1): -1]]
y_test = y[perm[int(0.75 * len(perm) + 1): -1]]
"""
def execute_ga(index):
    global x_train
    global y_train
    global x_test
    global y_test
    global n_inputs
    global n_classes
    global model
    if index == 0:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_iris_dataset()
    elif index == 1:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_wine_dataset()
    else:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_cancer_dataset()

    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    input_layer  = tensorflow.keras.layers.Input(n_inputs)
    leaky_relu = LeakyReLU(alpha=0.01)
    dense_layer1 = tensorflow.keras.layers.Dense(5, input_dim=n_inputs, activation="relu")(input_layer)
    #dense_layer2 = tensorflow.keras.layers.Dense(8, input_dim=4, activation="relu")(dense_layer1)
    #dense_layer3 = tensorflow.keras.layers.Dense(8, input_dim=4, activation="relu")(dense_layer2)

    output_layer = tensorflow.keras.layers.Dense(n_classes, activation="softmax")(dense_layer1)
    model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)
    keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=100)
    num_generations = 200
    num_parents_mating = 50
    initial_population = keras_ga.population_weights
    start_time = time.time()
    ga_instance = pygad.GA(num_generations=num_generations, 
                        num_parents_mating=num_parents_mating, 
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=callback_generation,
                        parent_selection_type= "rank",
                        crossover_type= "two_points",
                        mutation_type= "random")

    ga_instance.run()
    time_seconds = (time.time() - start_time)
    #print(ga_instance.best_solutions_fitness)
    #ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # Fetch the parameters of the best solution.
    best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                weights_vector=solution)
    model.set_weights(best_solution_weights)
    predictions = model.predict(x_train)
    # print("Predictions : \n", predictions)

    # Calculate the categorical crossentropy for the trained model.
    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    print("Categorical Crossentropy : ", cce(y_train, predictions).numpy())

    # Calculate the classification accuracy for the trained model.
    ca = tensorflow.keras.metrics.CategoricalAccuracy()
    ca.update_state(y_train, predictions)
    accuracy = ca.result().numpy()
    f1_train = (f1_score(y_train.argmax(1), predictions.argmax(1), average='macro'))
    prec_train = (precision_score(y_train.argmax(1), predictions.argmax(1), average='macro'))
    rec_train = (recall_score(y_train.argmax(1), predictions.argmax(1), average='macro'))

    print("Accuracy : ", accuracy)

    predictions = model.predict(x_test)
    # print("Predictions : \n", predictions)

    # Calculate the categorical crossentropy for the trained model.
    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    print("Test Categorical Crossentropy : ", cce(y_test, predictions).numpy())

    # Calculate the classification accuracy for the trained model.
    ca = tensorflow.keras.metrics.CategoricalAccuracy()
    ca.update_state(y_test, predictions)
    accuracy_test = ca.result().numpy()
    print("Test Accuracy : ", accuracy_test)
    f1_test = (f1_score(y_test.argmax(1), predictions.argmax(1), average='macro'))
    prec_test = (precision_score(y_test.argmax(1), predictions.argmax(1), average='macro'))
    rec_test = (recall_score(y_test.argmax(1), predictions.argmax(1), average='macro'))
    return time_seconds , ga_instance.best_solutions_fitness, accuracy_test, f1_test, prec_test, rec_test, accuracy, f1_train, prec_train, rec_train

#execute_ga()
