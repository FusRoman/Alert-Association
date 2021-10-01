import tensorflow as tf
from tensorflow.keras import backend as K

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
        # compute the loss of the model
        logits = model(x, training=True)
        loss_value = losses(y, logits)
    
    # compute the gradient respect to the loss of the model
    grads = tape.gradient(loss_value, model.trainable_variables)
    
    # applies the gradient to the optimizer and discard the None values.
    optimizer.apply_gradients((grad, var)
                           for (grad, var) in zip(grads, model.trainable_variables)
                           if grad is not None
                          )
    # update the states of the metrics to keep trace of model performance
    for metric in metrics:
        metric.update_state(y, logits)
    
    return loss_value

# Custom test step
@tf.function()
def test_step(model, losses, metrics, x, y):
    # do the same logics as the training function without gradient update
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