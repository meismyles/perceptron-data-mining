import time
import random
import numpy

def process(userChoice):
    bias = 0
    train_errors = []
    test_errors = []

    # Read input of iterations wanted
    maxIter = int(raw_input("\nHow many iterations would you like to run? "))

    # Get all features from files and determine the number of features
    featspace = get_feat_space("train.positive")
    featspace = featspace.union(get_feat_space("train.negative"))
    featspace = featspace.union(get_feat_space("test.positive"))
    featspace = featspace.union(get_feat_space("test.negative"))
    D = len(featspace)
    print "Dimensionality of the feature space: %i\n" % D

    # Convert feature set to a list
    featspace = list(featspace)

    ''' Create a feature index that stores each unique feature as {Key: Value}
        where 'Key' is a feature e.g. 'atmosphere' and 'Val' is an id (0,1,...,D)
        Start the feat_index at 1 since bias will be 0
    '''
    feat_index, weights = {}, {}
    for (feat_id, feat_val) in enumerate(featspace):
        feat_index[feat_val] = feat_id
        weights[feat_val] = 0

    # Generate the train and test data
    train_data = get_feat_vects("train.positive", D, 1)
    train_data.extend(get_feat_vects("train.negative", D, -1))
    test_data = get_feat_vects("test.positive", D, 1)
    test_data.extend(get_feat_vects("test.negative", D, -1))


    ''' Run different train/test methods depending on users choice.
        Choice 1 enters the if block and runs all training iterations before
        running a single test iteration.
        Choice 2 enters the else if block and runs a test iteration after every
        training iteration using the results to plot a graph of error rate.
    '''
    if userChoice == 1:
        startTime = time.clock()
        print "--------------------TRAINING-------------------"
        for i in range(maxIter):
            # Shuffle the 'train_data' and train the perceptron
            random.shuffle(train_data)
            weights, bias, train_errors = percept_train(i, D, train_data, weights, bias, train_errors)
        print "Overall Time taken: %.2f second(s)" % (time.clock() - startTime)

        print "--------------------TESTING--------------------"
        # Test the perceptron using the 'test_data'
        test_errors = percept_test(D, test_data, weights, bias, test_errors)

    elif userChoice == 2:
        for i in range(maxIter):
            print "- RUN %i - - - - - - - - - - - - - - - - - - - -" % (i+1)
            # Shuffle the 'train_data' and train the perceptron
            random.shuffle(train_data)
            weights, bias, train_errors = percept_train(i, D, train_data, weights, bias, train_errors)

            # Shuffle the 'test_data' and test the perceptron
            random.shuffle(test_data)
            test_errors = percept_test(D, test_data, weights, bias, test_errors)

        # Plot a graph of error rate vs number of iterations
        plot_graph(train_errors, test_errors, maxIter)


# Method to determine the feature space from some input files
def get_feat_space(fname):
    feats = set()
    with open(fname) as feat_file:
        for line in feat_file:
            for w in line.strip().split():
                feats.add(w)
    return feats


# Method to determine a feature vector for each input file
def get_feat_vects(fname, D, label):
    feat_vects = []
    with open(fname) as feat_file:
        for line in feat_file:
            feats = {}
            for w in line.strip().split():
                feats[w] = 1
            feat_vects.append((feats,label))
    return feat_vects


# Method to train the perceptron
def percept_train(i, D, train_data, weights, bias, train_errors):
    startTime = time.clock()
    numErrors = 0

    for review in train_data:
        feat_set, expected = review[0], review[1]
        activation = sum(weights[feat]*feat_set[feat] for feat in feat_set) # calculate the activation
        activation += bias # adding the bias

        # If output was incorrect, update the weights
        if ((expected*activation) <= 0):
            numErrors += 1
            for feat in feat_set:
                weights[feat] = weights[feat] + expected*feat_set[feat]
            bias += expected # updating the bias

    # Add error data to an array to be used later
    train_error = (float(numErrors)/float(len(train_data)))*100
    train_errors.append(train_error)

    print "Training iteration %i. Time taken: %.2f second(s)" % ((i+1), (time.clock() - startTime))

    return (weights, bias, train_errors)


# Method to test the perceptron
def percept_test(D, test_data, weights, bias, test_errors):
    totalCorrect = 0
    numErrors = 0

    for review in test_data:
        feat_set, expected = review[0], review[1]
        activation = sum(weights[feat]*feat_set[feat] for feat in feat_set) # calculate the activation
        activation += bias # adding the bias

        # Store the number of correct vs incorrect outputs
        if (numpy.sign(activation) == expected):
            totalCorrect += 1
        else:
            numErrors += 1

    # Add error data to an array to be used later
    test_error = (float(numErrors)/float(len(test_data)))*100
    test_errors.append(test_error)

    correctPercent = (float(totalCorrect)/len(test_data))*100
    print "Correctly classified instances: %i/%i (%.1f%%)" % (totalCorrect, len(test_data), correctPercent)
    print "Error rate in classification: %.1f%%" % (100-correctPercent)
    print "-----------------------------------------------\n"

    return test_errors


# Method to plot a graph of error rate vs number of iterations
def plot_graph(train_errors, test_errors, maxIter):
    from matplotlib import pyplot as plt
    plt.plot(range(1, maxIter+1), train_errors, '-b', label='Training Errors')
    plt.plot(range(1, maxIter+1), test_errors, '-r', label='Testing Errors')
    plt.axis([1,maxIter,0,(max(max(train_errors), max(test_errors))+5)])
    plt.xlabel('Iteration Number')
    plt.ylabel('Error Rate %')
    plt.legend(loc='upper right')
    plt.show(block=True)


#Execute
if __name__ == "__main__":
    print "\nPlease choose one of the following options:"
    print "1. Repeat training multiple times and then perform single test"
    print "2. Test after each iteration of training and output plot of results (requires matplotlib)"
    userChoice = int(raw_input())
    process(userChoice) # execute the main method
