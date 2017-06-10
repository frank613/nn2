#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    mylogisticClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nLogsticregression has been training..")
    mylogisticClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = mylogisticClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\n Result of the Logsticregression recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)
    for i in range(2):
        for j in range(2):
            learningRate = (i+1)*0.002
            epochs = (j+1)*20


            mylogisticClassifier = LogisticRegression(data.trainingSet,
                                                  data.validationSet,
                                                  data.testSet,
                                                  learningRate=learningRate,
                                                  epochs=epochs)

            # Train the classifiers
            print("=========================")
            print("learning rate :" + str(learningRate))
            print("epoch :" + str(epochs))
            print("Training..")


            print("\nLogsticregression has been training..")
            mylogisticClassifier.train()
            print("Done..")

            # Do the recognizer
            # Explicitly specify the test set to be evaluated
            perceptronPred = mylogisticClassifier.evaluate()

            # Report the result
            print("=========================")
            evaluator = Evaluator()



            print("\n Result of the Logsticregression recognizer:")
            # evaluator.printComparison(data.testSet, perceptronPred)
            evaluator.printAccuracy(data.testSet, perceptronPred)
    
    
if __name__ == '__main__':
    main()
