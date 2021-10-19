# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:09:51 2021

@author: Roman
"""

from tensorflow.keras.models import Model
from spektral.models.general_gnn import MLP
from src.model.motlayer import MOTLayer

class MOTModel(Model):
    """
    Build a MOTModel which consist of one or multiple MOTLayer and one MLP Layer used to perform final edge classification
    """
    def __init__(
        self,
        node_inputs_channels,
        edge_inputs_channels,
        edge_hidden=2,
        edge_layers=2,
        node_hidden=3,
        node_layers=2,
        flow_in_hidden=3,
        flow_in_layers=2,
        flow_out_hidden=3,
        flow_out_layers=2,
        mp_activation=None,
        mp_final_activation=None,
        edge_classifier_hidden=2,
        edge_classifier_layer=2,
        edge_classifier_activation="prelu",
        edge_classifier_final_activation="sigmoid",
        message_passing=4,
        batch_norm=True,
        dropout=0.0,
    ):
        """
        MOTModel Constructor

        Parameters
        ----------

        node_inputs_channels : integer
            number of nodes feature
        edge_inputs_channels : integer
            number of edges feature
        edge_hidden : integer 
            number of units of the edge mlp
        edge_layer : integer
            number of layers of the edge mlp
        node_hidden : integer 
            number of units of the node mlp
        node_layer : integer
            number of layers of the node mlp
        flow_in_hidden : integer 
            number of units of the flow_in mlp
        flow_in_layer : integer
            number of layers of the flow_in mlp
        flow_out_hidden : integer 
            number of units of the flow_out mlp
        flow_out_layer : integer
            number of layers of the flow_out mlp
        mp_activation : string or keras activation class
            the activation function of the hidden layer of the four mlp in the motlayer
        mp_final_activation : string or keras activation class
            the activation function of the last mlp layer of the four mlp in the motlayer
        edge_classifier_hidden : integer
            number of unit of the edge_classifier
        edge_classifier_layer : integer
            number of layer of the edge classifier
        edge_classifier_activation : string or keras activation class
            the activation function of the hidden layer of the edge_classifier
        edge_classifier_final_activation : string or keras activation class
            the activation function of the last layer of the edge_classifier
        message_passing : integer
            number of message passing step, build a number of message passing layer equal to this argument
        batch_norm : Boolean
            if is true, used a BatchNormalization Layer in the edge_classifier
        dropout : float between 0 and 1
            fraction of the input units to drop        
        """
        super().__init__()
        
        self.edge_mlp = MLP(edge_inputs_channels, 
                     hidden=edge_hidden,
                     layers=edge_layers,
                     activation=mp_activation,
                     final_activation=mp_final_activation)
        
        self.flow_in = MLP(node_inputs_channels, 
                           hidden=flow_in_hidden,
                           layers=flow_in_layers,
                           activation=mp_activation,
                           final_activation=mp_final_activation)
        
        self.flow_out = MLP(node_inputs_channels,
                            hidden=flow_out_hidden,
                            layers=flow_out_layers,
                            activation=mp_activation,
                            final_activation=mp_final_activation)
        
        self.node_mlp = MLP(node_inputs_channels,
                             hidden=node_hidden,
                             layers=node_layers,
                             activation=mp_activation,
                             final_activation=mp_final_activation)
        
        # Attention, prelu ne fonctionne pas en tant qu'activation pour les couches MOTMPN
        self.mot_gnn = [
            MOTLayer(edge_inputs_channels, self.edge_mlp, self.flow_in, self.flow_out, self.node_mlp)
            for _ in range(message_passing)
        ]

        self.edge_classifier = MLP(
            1,
            edge_classifier_hidden,
            edge_classifier_layer,
            batch_norm,
            dropout,
            activation=edge_classifier_activation,
            final_activation=edge_classifier_final_activation,
        )

    def call(self, inputs, training=None):
        """
        Function applied for a MOTModel call, perform all the message passing steps then return edge prediction based on the edge_classifier.

        Parameters
        ----------
        inputs : graph batch information
            node feature, adjacendy matrix, edge feature and the index that locate the past edges and futur edges for each nodes contains in one batch
        training : Boolean
            set the training mode, impact only BatchNormalizationLayer
        
        Return
        ------
        edge_classifier_prediction : numpy array or tensor
            the edge classifier prediction over all edges
        """
        if len(inputs) == 5:
            node_out, a, edge_out, past_index, futur_index = inputs
        else:
            raise ValueError
            
        # Message passing
        for layer in self.mot_gnn:
            layer_input = [
                node_out,
                a,
                edge_out,
                past_index,
                futur_index
            ]
            node_out, edge_out = layer(layer_input, training=training)

        # Edge classifier
        return self.edge_classifier(edge_out, training=training)
    
    
