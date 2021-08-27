# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 15:51:21 2021

@author: Roman
"""

from spektral.data.graph import Graph


class MOTGraph(Graph):
    def __init__(self, x, a, e, y, edge_info, past_index, futur_index, **kwargs):
        """
        Build a graph based on Spektral package. Simply extend graph definition of Spektral to add past_index and futur_index to keep temporal information.
        Keep information about alerts used to build the graph in order to create trajectories
        
        Parameters
        ----------
        x : numpy array
            node feature of graph
        a : scipy sparse matrix
            sparse adjacency matrix of graph
        e : numpy array
            edge feature of graph
        y : numpy array
            label used to train graph neural network
        edge_info : dataframe
            dataframe containing information (candid, nid, ssnamenr, label) about alerts that have been used to build the graph
        past_index : numpy array
            index of all edges from the past of all node
        futur_index : numpy array 
            index of all edges from the futur of all node
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.past_index = past_index
        self.futur_index = futur_index
        self.edge_info = edge_info
        super().__init__(
            x=x, 
            a=a, 
            e=e, 
            y=y, 
            **kwargs)


if __name__ == "__main__":
    print("test")

