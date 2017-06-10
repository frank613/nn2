# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=False):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0
        yzhou = []
        while not learned:
            totalError = 0

            grad = np.zeros(self.trainingSet.input[0].shape)
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)

                error = loss.calculateError(label, output)
                grad=grad+error*input
                if label==1 and output < 0.5:
                    totalError += 1
                if label==0 and output >= 0.5:
                    totalError += 1

            yzhou.append(totalError)
            self.updateWeights(grad)
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)

            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                xzhou=np.arange(iteration)
                plt.plot(xzhou,np.array(yzhou))
                plt.title("Lerning rate %f"%self.learningRate);
                plt.xlabel('epoch')
                plt.ylabel('Error')
                plt.show()
                learned = True
        pass
        
    def classify(self, testInstance):
        """Classify a single instance.
f
        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) >= 0.5
        pass

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight += self.learningRate * grad
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
