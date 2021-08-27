# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:05:46 2021

@author: Roman
"""

from spektral.layers import MessagePassing
from tensorflow.keras import backend as K
import tensorflow as tf

class MOTLayer(MessagePassing):
    def __init__(self, 
                 n_edges_features,
                 edge_mlp, 
                 flow_in_mlp, 
                 flow_out_mlp, 
                 node_mlp, 
                 aggregate="sum"):
        super().__init__(aggregate=aggregate)
        
        self.n_edges_features = n_edges_features
        
        self.edge_mlp = edge_mlp

        self.flow_in = flow_in_mlp

        self.flow_out = flow_out_mlp

        self.node_mlp = node_mlp
          
    def call(self, data, training=None):
        x, a, e, past_index, futur_index = data
        
        self.past_index = past_index
        self.futur_index = futur_index
        

        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        source_node = self.get_i(x)
        source_target = self.get_j(x)
        edge_embedding = [source_node, source_target, e]
        
        # Edge update
        edges_output = self.edge_mlp(K.concatenate(edge_embedding), training=training)
        nodes_output = self.propagate(x, a, edges_output, training=training)
        return nodes_output, edges_output

    def propagate(self,x, a, e, training=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]

        # Message
        past_message, futur_message = self.message(x, e, training=training)

        # Aggregate
        past_embedding = self.aggregate(past_message)
        futur_embedding = self.aggregate(futur_message)

        # Update
        output = self.update((past_embedding, futur_embedding), training=training)

        return output
    
    def message(self,x, e, training):
        # Get the node features of all neighbors
        node_i = self.get_i(x)
        
        # get the edges from the past of node_i
        past_edge = tf.scatter_nd(
            self.past_index, 
            tf.gather(e, self.past_index)[:, 0, :], 
            (len(e), self.n_edges_features)
        )
        
        # get the edges from the futur of node_i
        futur_edge = tf.scatter_nd(
            self.futur_index, 
            tf.gather(e, self.futur_index)[:, 0, :], 
            (len(e), self.n_edges_features)
        )
        
        past_arg = [node_i, past_edge]
        futur_arg = [node_i, futur_edge]
        
        past_embedding = self.flow_in(K.concatenate(past_arg), training=training)
        futur_embedding = self.flow_out(K.concatenate(futur_arg), training=training)

        return past_embedding, futur_embedding

    def update(self, embeddings, training=None):
        past_embedding, futur_embedding = embeddings
        
        # Node update
        return self.node_mlp(K.concatenate([past_embedding, futur_embedding]), training=training)
    
    

    
    
