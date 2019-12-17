import numpy as np

####### INPUT ########
# X = np.array([[1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 2, 3, 4, 5]]) #row vector a = np.array([[1,2,3]]) # a 'row vector' b = np.array([[1],[2],[3]]) # a 'column vector'
inputData = np.array([[1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 7, 8, 3], [1, 3, 4, 5, 4, 3, 2, 1, 4, 5, 5, 4, 5]])
outputData = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
alpha = 0.1
iterations = 10000

######### TEST ##############
test = np.array([[9], [5]])


#############################

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def nn(n_h, X, Y, learning_rate, num_iterations, print_cost=False):
    ## Assign Dimensions ##
    n_x = X.shape[0]
    n_y = Y.shape[0]
    m = X.shape[1]
    z1 = np.zeros((n_h, m))
    z2 = np.zeros((1, m))

    ## Random Initialization of Parameters ##
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, w1.shape[0]) * 0.01
    b2 = np.zeros((n_y, 1))
    print(w2.shape)

    for i in range(0, num_iterations):

        ## Forward Propagation ##
        z1 = np.dot(w1, X) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)

        ## Cost ##
        cost = -1 / m * np.sum(Y * np.log(a2) + (1 - Y) * np.log(1 - a2))

        ## Backward propagation
        dz2 = a2 - Y
        dw2 = 1 / m * np.dot(dz2, a1.T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
        dw1 = 1 / m * np.dot(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        ## Update Parameters ##
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    prams = {"w1": w1,
             "b1": b1,
             "w2": w2,
             "b2": b2}

    return prams


def predict(prams, X):
    w1 = prams["w1"]
    b1 = prams["b1"]
    w2 = prams["w2"]
    b2 = prams["b2"]

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    return a2


if __name__ == '__main__':
    parameters = nn(5, inputData, outputData, alpha, iterations, True)
    predictions = predict(parameters, inputData)
    predictions = 1 * (predictions > 0.5)
    print('Accuracy: %d' % float(
        (np.dot(outputData, predictions.T) + np.dot(1 - outputData, 1 - predictions.T)) / float(
            outputData.size) * 100) + '%')

    test_predict = predict(parameters, test)
    prob = float(test_predict) * 100
    print("\nThe Probability of the result being positive is %.1f" % prob + "%")
