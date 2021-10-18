# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:20:32 2021

@author: Roman
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from keras import metrics
from src.model.motmodel import MOTModel
from src.graph.motloader import MOTLoader
import time as t
from spektral.transforms import AdjToSpTensor
from spektral.transforms.normalize_one import NormalizeOne
from src.graph.motgraphdataset import MOTGraphDataset, EdgeNormalizeOne
import numpy as np
import matplotlib.pyplot as plt
import json

# Loss
def weighted_binary_cross_entropy(y_true, y_pred):
    nb_ones = tf.math.reduce_sum(y_true)    
    nb_zeros = tf.shape(y_true)[0] - nb_ones
    ones_prop = nb_zeros // nb_ones
    weights = (y_true * (ones_prop - 1)) + 1
    
    bce = K.binary_crossentropy(tf.cast(y_true, tf.float32), y_pred)
    weighted_bce = K.mean(bce * tf.cast(weights, tf.float32))
    return weighted_bce

# Custom train step
@tf.function()
def train_step(model, optimizer, losses, metrics, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = losses(y, logits)
        
    grads = tape.gradient(loss_value, model.trainable_variables)
    
    optimizer.apply_gradients((grad, var) 
                           for (grad, var) in zip(grads, motmodel.trainable_variables)
                           if grad is not None
                          )
    
    for metric in metrics:
        metric.update_state(y, logits)
    
    return loss_value

# Custom test step
@tf.function()
def test_step(model, losses, metrics, x, y):
    logits = model(x, training=False)
    loss_value = losses(y, logits)
    for metric in metrics:
        metric.update_state(y, logits)
    return loss_value
        
def make_inputs(data, graph_i):
    x = data[0][graph_i]
    a = data[1][graph_i]
    e = data[2][graph_i]
    past_index = data[3][graph_i]
    futur_index = data[4][graph_i]
    return [x, a, e, past_index, futur_index]

def manage_metrics(metric, result_dict):
    all_res = result_dict.setdefault(metric.name, [])
    result = metric.result()
    print("\t\t{} : {}".format(metric.name, result))
    all_res.append(result)
    result_dict[metric.name] = all_res
    metric.reset_states()
    return result_dict

def print_current_graph(step, max_samples):
    if step == 0:
        print("graph {}".format(max_samples))
    else:
        print("graph {}".format(step))

if __name__=="__main__":
    # Optimizers
    o_sgd = optimizers.SGD(learning_rate=1e-3)
    o_adam = optimizers.Adam()
    o_adadelta = optimizers.Adadelta()
    
    # losses = BinaryCrossentropy()
    losses = weighted_binary_cross_entropy
    
    # Metrics
    training_metrics = [
        metrics.BinaryAccuracy(name="ba"),
        metrics.FalseNegatives(name="fn"),
        metrics.FalsePositives(name="fp"),
        metrics.TrueNegatives(name="tn"),
        metrics.TruePositives(name="tp"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall")
    ]
    
    validation_metrics = [
        metrics.BinaryAccuracy(name="ba"),
        metrics.FalseNegatives(name="fn"),
        metrics.FalsePositives(name="fp"),
        metrics.TrueNegatives(name="tn"),
        metrics.TruePositives(name="tp"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall")
    ]
    
    params_file = open("training_params.json")
    data_params = json.load(params_file)
    
    # Params
    EPOCHS = data_params['EPOCHS']
    verbose = data_params['VERBOSE']
    
    t_before = t.time()
    train_window = (data_params['DATASET_PARAMS']['train_dataset_params']['window_size'],
                    data_params['DATASET_PARAMS']['train_dataset_params']['window_overlap'])
    tr_dataset = MOTGraphDataset(data_params['DATASET_PARAMS']['train_dataset_params']['month'], 
                                 data_params['DATASET_PARAMS']['train_dataset_params']['type'],
                                 data_params['DATASET_PARAMS']['train_dataset_params']['point_limit'],
                                 window_params=train_window,
                                transforms=[EdgeNormalizeOne(), NormalizeOne(), AdjToSpTensor()])
    print("tr_dataset construct time: ", t.time() - t_before)
    
    t_before = t.time()
    valid_window = (data_params['DATASET_PARAMS']['valid_dataset_params']['window_size'],
                    data_params['DATASET_PARAMS']['valid_dataset_params']['window_overlap'])
    val_dataset = MOTGraphDataset(data_params['DATASET_PARAMS']['valid_dataset_params']['month'], 
                                 data_params['DATASET_PARAMS']['valid_dataset_params']['type'],
                                 data_params['DATASET_PARAMS']['valid_dataset_params']['point_limit'],
                                 window_params=valid_window,
                                transforms=[EdgeNormalizeOne(), NormalizeOne(), AdjToSpTensor()])
    print("dataset construct time: ", t.time() - t_before)
    
    # Model and Loader Initialisation
    motmodel = MOTModel(tr_dataset.n_node_features,
                        tr_dataset.n_edge_features,
                        edge_layers=data_params['MODEL_PARAMS']['edge_layers'],
                        edge_hidden=data_params['MODEL_PARAMS']['edge_hidden'],
                        node_layers=data_params['MODEL_PARAMS']['node_layers'],
                        node_hidden=data_params['MODEL_PARAMS']['node_hidden'],
                        flow_in_layers=data_params['MODEL_PARAMS']['flow_in_layers'],
                        flow_in_hidden=data_params['MODEL_PARAMS']['flow_in_hidden'],
                        flow_out_layers=data_params['MODEL_PARAMS']['flow_out_layers'],
                        flow_out_hidden=data_params['MODEL_PARAMS']['flow_out_hidden'],
                        edge_classifier_layer=data_params['MODEL_PARAMS']['edge_classifier_layer'],
                        message_passing=data_params['MODEL_PARAMS']['MP_STEPS'])
    
    loader_tr = MOTLoader(tr_dataset, epochs=EPOCHS, batch_size=1)
    loader_va = MOTLoader(val_dataset, epochs=EPOCHS, batch_size=1)
    
    
    # Custom training loop
    tr_epoch_steps = 1
    current_epoch = 0
    
    tr_epoch_loss = []
    tr_loss = []
    va_loss = []
    tr_metrics_results = dict()
    va_metrics_results = dict()
    
    print("CURRENT EPOCH: {}, REMAIN: {}".format(current_epoch, EPOCHS - current_epoch))
    print("First epoch take longer than the following")
    
    for tr_batch in loader_tr:
        
        
        
        tr_step = tr_epoch_steps % loader_tr.steps_per_epoch
        if verbose == "True":
            print_current_graph(tr_step, loader_tr.steps_per_epoch)
        
        if tr_step == 1:
            tr_before = t.time()
            
        for graph_i in range(loader_tr.batch_size):
            tr_data = tr_batch[0]
            tr_label = tr_batch[1][graph_i]
            
            tr_inputs = make_inputs(tr_data, graph_i)
            current_tr_loss = train_step(motmodel, o_adam, losses, training_metrics, tr_inputs, tr_label)
            tr_epoch_loss.append(current_tr_loss)
    
            
        tr_epoch_steps += 1
        
        if tr_step == 0:
            print("DISPLAY TRAINING RESULTS:")
            epoch_loss = np.mean(tr_epoch_loss)
            print("\ttraining_mean_loss: {}".format(epoch_loss))
            tr_epoch_loss = []
            tr_loss.append(epoch_loss)
            
            print()
            print("\ttraining metric results:")
            for tr_metric in training_metrics:
                tr_metrics_results = manage_metrics(tr_metric, tr_metrics_results)
            
            
            val_epoch_loss = []
            val_epoch_steps = 1
            
            for va_batch in loader_va:
                val_step = val_epoch_steps % loader_va.steps_per_epoch
                
                if verbose == 'True':
                    print_current_graph(val_step, loader_va.steps_per_epoch)
                
                if val_step == 1:
                    va_before = t.time()
                
                for graph_i in range(loader_va.batch_size):
                    val_data = va_batch[0]
                    val_label = va_batch[1][graph_i]
    
                    val_inputs = make_inputs(val_data, graph_i)
                    current_val_loss = test_step(motmodel, losses, validation_metrics, val_inputs, val_label)
                    val_epoch_loss.append(current_val_loss)
                
                val_epoch_steps += 1
                
                if val_step == 0:
                    print("\tEND VALIDATION STEP, elapsed time: {} sec".format(t.time() - va_before))
                    print()
                    print("\tDISPLAY VALIDATION RESULTS")
                    
                    epoch_loss = np.mean(val_epoch_loss)
                    print("\tvalidation_mean_loss: {}".format(epoch_loss))
                    val_epoch_loss = []
                    va_loss.append(epoch_loss)
                    for va_metric in validation_metrics:
                        va_metrics_results = manage_metrics(va_metric, va_metrics_results)
                    break
                
            print("END EPOCH, elapsed time: {} sec".format(t.time() - tr_before))
            print()
            print()
            current_epoch += 1
            print("CURRENT EPOCH: {}, REMAIN: {}".format(current_epoch, EPOCHS - current_epoch))
    
    print()
    
    plt.plot(np.arange(len(tr_loss)), tr_loss, label="tr_loss")
    plt.plot(np.arange(len(va_loss)), va_loss, label="va_loss")
    plt.legend()
    plt.savefig("training_result/train_val_loss")
    plt.close()
    
    for tr_metric_name, tr_metric_result in tr_metrics_results.items():
        plt.plot(np.arange(len(tr_metric_result)), tr_metric_result, label="tr_" + tr_metric_name)
        plt.legend()
        plt.savefig("training_result/train_metric_" + tr_metric_name)
        plt.close()
    
    for va_metric_name, va_metric_result in va_metrics_results.items():
        plt.plot(np.arange(len(va_metric_result)), va_metric_result, label="va_" + va_metric_name)
        plt.legend()
        plt.savefig("training_result/valid_metric_" + va_metric_name)
        plt.close()
    
    