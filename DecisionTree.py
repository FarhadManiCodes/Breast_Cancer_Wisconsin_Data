""" Decision Tree Classifier"""
from typing import Tuple, Union
import pandas as pd
import numpy as np
from math import log2
from dataclasses import dataclass

@dataclass
class threshold_result:
    index:int
    measure:float

def entropy(labels):
    """ Computes entropy of label distribution. """
    p = labels.sum()/len(labels)
    if 0 < p < 1:
        return -p * log2(p) - (1 - p) * log2(1 - p)
    else:
        return 0     # Compute entropy


def findtreshold(y_sorted:np.ndarray,criterion:str) -> threshold_result:
    if criterion == 'entropy':
        min_entropy = 1
        threshold_index = 1
        label_size=len(y_sorted)
        for i in range(1,label_size):
            ent = ((i) * entropy(y_sorted[:i]) +
                   (label_size-i) * entropy(y_sorted[i:]))/label_size
            if ent < min_entropy:
                min_entropy = ent
                threshold_index = i

        return threshold_result(threshold_index, min_entropy)


class Node:
    """
    Class for each node of the Decision Tree
    """

    def __init__(self,
                X:np.ndarray,
                y:np.ndarray,
                *,criterion = "entropy"):
        self.X = X
        self.y = y
        self.criterion = criterion
        self.processed = False

    def __str__(self) -> str:
        if self.processed:
            return f"""Node object Processed: 
            - current {self.criterion} : {self.entropy}
            - best column for spliting: {self.best_col}
            - threshold : {self.threshold}
            - {self.criterion} after the split : {self.measure}
                    """
        else:
            return "Unprocessed Node"

    def find_the_best_feature(self):
        min_measure = 1.0
        best_col = None
        threshold = None
        if self.criterion == "entropy":
            self.entropy = entropy(self.y)
        if self.entropy > 0:
            for i,col in enumerate(self.X.T):
                sorted_ind = np.argsort(col)
                y_sorted = self.y[sorted_ind]
                result= findtreshold(y_sorted,self.criterion)
                if result.measure < min_measure:
                    min_measure = result.measure
                    best_col = i
                    threshold = (col[sorted_ind][result.index-1] +
                                col[sorted_ind][result.index]) / 2
        self.best_col = best_col
        self.threshold = threshold
        self.measure = min_measure
        self.processed = True

class DecisionTreeClassifier:
    """
    A decision tree classifier.
    """
    
    def __init__(self,*,
                criterion:str="entropy",
                max_depth:int=None,
                min_sample_split:int=2):
        """ 
        Constructor: 

        Parameters:
        -----------
        criterion: {"gini","entropy"}, default="entropy"
            The function to measure the quality of a split.
            "gini" for Gini impurity and "entropy" for information gain.

        max_depth : int, default=None,
            The maximum depth of the tree. If None, the nodes are expanded until
            all leaves are pure of until all leaves contain less than
            min_sample_split split samples.

        min_sample_split: int, default=2
            The minimum number of samples required to split an internal node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

    def __str__(self) -> str:
        text=f"""DecisionTreeClassifier(criterion = {self.criterion},
                        max_depth = {self.max_depth},
                        min_sample_split = {self.min_sample_split})
            """
        return text

    def fit(self,
            X:pd.DataFrame,
            y:np.ndarray):
        depth = 0
        if self.max_depth:
            max_depth = self.max_depth
        else:
            max_depth = int(log2(len(y)))
        while (depth < max_depth):
            N = Node(X.to_numpy(),y)
            N.find_the_best_feature()
            print(N)
            depth += 1