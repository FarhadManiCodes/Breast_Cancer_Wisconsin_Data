""" Decision Tree Classifier"""
from typing import Union
import pandas as pd
import numpy as np
from math import log2


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

    def __str__(self) -> str:
        return "Node (class)"

    def find_the_best_feature(self):
        for col in range(self.X.shape[1]):
            pass       



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
            y:Union[np.ndarray,pd.Series]):
        depth = 0
        if self.max_depth:
            max_depth = self.max_depth
        else:
            max_depth = int(log2(len(y)))
        while (depth < max_depth):
            print('test')
            depth += 1