import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.ids = (1, 2)
        
        self.ws = np.array([], dtype=np.float32)
        self.labels = np.array([], dtype=np.uint8)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        self.labels = np.unique(y)
        K = len(self.labels)  # for convenience
        X_b = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # append 1 to every point for the bias
        self.ws = np.zeros((K, X_b.shape[1]), dtype=np.float32)  # initialize all wi vectors
        
        finished = False
        while not finished:
            finished = True  # if there will be no changes in self.ws then it will remain True and we will finish
            for i, xi in enumerate(X_b):
                argmax_index = self.predict_single(xi)
                y_pred = self.labels[argmax_index]
                if y_pred != y[i]:
                    self.ws[np.where(self.labels == y[i])] += xi
                    self.ws[argmax_index] -= xi
                    finished = False
        return True  # assuming the data is always linear separable
    
    
    def predict_single(self, x):
        """
        Return an argmax label (the first argmax if there's several) 
        for single vector x (ws must be calculated first in the fit method).
        """
        return self.labels[np.argmax(np.array([wi @ x for wi in self.ws]))]
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        pred_y = lambda x: self.labels[self.predict_single(x)]  # this will return the predicted label for x
        X_b = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1) # X + bias term entry for each point xi
        
        return np.array([pred_y(x) for x in X_b]) # list of predicted labels for X


if __name__ == "__main__":
    
    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
