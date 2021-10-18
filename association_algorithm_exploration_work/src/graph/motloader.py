# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:11:34 2021

@author: Roman
"""

from spektral.data.loaders import BatchLoader
import numpy as np
from src.graph.motgraphdataset import MOTGraphDataset
from spektral.transforms.normalize_one import NormalizeOne
from src.graph.motgraphdataset import EdgeNormalizeOne
from spektral.transforms import AdjToSpTensor

class MOTLoader(BatchLoader):
    
    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        """
        Build a motloader from a dataset.

        Parameters
        ----------
        dataset : motgraphdataset
            a dataset of motgraph
        batch_size : integer
            used to specify the number of graph in a batch.
            A batch is a set of data from the dataset where the model
            make prediction over all data and compute the loss.
            The models wheights are update only at the end of a batch.
        epochs : integer
            The number of time that the training iterate over the full dataset.
        shuffle : boolean
            if setting to true, shuffle the dataset before training.
            No shuffle otherwise.

        """
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    
    def collate(self, batch):
        """
        Construct the data structure from batch used by the multiple object tracking (mot) model.
        This function is used internally by BatchLoader from Spektral and work only if batch is set to 1 in the class constructor of MOTLoader.

        Parameters
        ----------
        batch : list
            contains all the data of the batch. If batch is set to 10 in the constructor of MOTLoader then the batch parameters of this function
            contains 10 data from the dataset.

        Return
        ------
        output : list
            a list of all available feature provided by the data.

        y : integer list
            label associated with the batch data.
        """
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch])
        e = np.array([g.e for g in batch])
        past_index = np.array([g.past_index for g in batch])
        futur_index = np.array([g.futur_index for g in batch])
        y = np.array([g.y for g in batch]) if "y" in self.dataset.signature else None
        
        output = [x, a, e, past_index, futur_index]
        
        if y is None:
            return output
        else:
            return output, y

if __name__=="__main__":

    tr_dataset = MOTGraphDataset("../../../data/month=03", 'Solar System MPC', 18, window_params=(5, 2),
                                transforms=[EdgeNormalizeOne(), NormalizeOne(), AdjToSpTensor()])
    

    print("instantiate the loader with the graph dataset")
    tr_loader = MOTLoader(tr_dataset, epochs=10)

    print("iterate over the loader for 10 epochs, each epochs contains {} graphs.".format(len(tr_dataset)))
    epochs_ = 0
    for tr_batch in tr_loader:
        if epochs_ % 8 == 0:
            print("epoch number : {}".format(epochs_/8))
        epochs_ += 1
        print("number of graphs seen : {}".format(epochs_))