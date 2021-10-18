from os import lseek, name
import numpy as np
import tensorflow

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.backend import sigmoid
import tensorflow_probability as tfp
from tensorflow import transpose as tf_transpose
from tensorflow import linalg
from tensorflow import sqrt as tf_sqrt
from tensorflow.keras.layers import BatchNormalization

def euclidean_distance(vectors):
	(featsA, featsB) = vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def build_LSTM(
     input_shape,
     nb_lstm_layers=1, 
     nb_lstm_units=16, 
     use_dropout=True,
     dropout=0.2,
     use_activation=True,
     activation = 'relu',
     last_dense_layer = False,
     name=''
     ):

    inputs = Input((None, input_shape))
    return_sequence = False
    if nb_lstm_layers > 1:
        return_sequence = True
    lstm_layers = LSTM(nb_lstm_units, return_sequences=return_sequence)(inputs)

    if nb_lstm_layers > 1:
        if use_dropout:
            lstm_layers = Dropout(dropout)(lstm_layers)
        if use_activation:
            lstm_layers = Activation(activation)(lstm_layers)

    for i in range(nb_lstm_layers-1):
        if i == nb_lstm_layers-2:
            return_sequence = False
            use_activation = False
            use_dropout = False
        lstm_layers = LSTM(nb_lstm_units, return_sequences=return_sequence)(lstm_layers)
        
        if use_dropout:
            lstm_layers = Dropout(dropout)(lstm_layers)
        if use_activation:
            lstm_layers = Activation(activation)(lstm_layers)

    if last_dense_layer:
        outputs = Dense(nb_lstm_units)(lstm_layers)
    else:
        outputs = lstm_layers

    return Model(inputs, outputs, name=name)

def build_GRU(
     input_shape, 
     nb_gru_layers=1, 
     nb_gru_units=16, 
     use_dropout=True,
     dropout=0.2,
     use_activation=True,
     activation = 'relu',
     last_dense_layer = False,
     name=''
     ):

    inputs = Input((None, input_shape))
    return_sequence = False
    if nb_gru_layers > 1:
        return_sequence = True
    gru_layers = GRU(nb_gru_units, return_sequences=return_sequence)(inputs)

    if nb_gru_layers > 1:
        if use_dropout:
            gru_layers = Dropout(dropout)(gru_layers)
        if use_activation:
            gru_layers = Activation(activation)(gru_layers)

    for i in range(nb_gru_layers-1):
        if i == nb_gru_layers-2:
            return_sequence = False
            use_activation = False
            use_dropout = False
        gru_layers = GRU(nb_gru_units, return_sequences=return_sequence)(gru_layers)
        
        if use_dropout:
            gru_layers = Dropout(dropout)(gru_layers)
        if use_activation:
            gru_layers = Activation(activation)(gru_layers)

    if last_dense_layer:
        outputs = Dense(nb_gru_units)(gru_layers)
    else:
        outputs = gru_layers

    return Model(inputs, outputs, name=name)

def build_dense(
    input_shape,
    nb_dense_layer=1, 
    nb_unit=16, 
    reduce_units = False,
    use_dropout=True, 
    dropout=0.2,
    use_activation=True,
    activation='relu',
    name=''
    ):

    def apply_reduce_units(units):
        if reduce_units:
            units /= 2
            if units < 1:
                return 1
            else:
                return units
        else:
            return units

    inputs = Input(input_shape)
    if nb_dense_layer == 1:
        print("nb_units not used because nb_dense_layer == 1")
        outputs = Dense(nb_unit)(inputs)
        return Model(inputs, outputs, name=name)
    else:
        dense_layers = Dense(nb_unit)(inputs)
        nb_unit = apply_reduce_units(nb_unit)

        if use_dropout:
            dense_layers = Dropout(dropout)(dense_layers)
        if use_activation:
            dense_layers = Activation(activation)(dense_layers)

        for i in range(nb_dense_layer-1):
            dense_layers = Dense(nb_unit)(dense_layers)
            nb_unit = apply_reduce_units(nb_unit)

            if i != nb_dense_layer - 2:
                if use_dropout:
                    dense_layers = Dropout(dropout)(dense_layers)
                if use_activation:
                    dense_layers = Activation(activation)(dense_layers)

        return Model(inputs, dense_layers, name=name)


def similarity_measure_model(
    nb_feature,
    distance_metric=None,
    nb_rrn_layer=1,
    nb_dense_layer=1,
    nb_final_layer=1,
    siamese_unit=16,    
    final_unit=32,
    reduce_unit_final=True,
    use_dropout=True,
    dropout=0.2,
    use_activation=True,
    activation='relu',
    batch_normalisation = True,
    last_rnn_dense_layer=False,
    RNN_layer = 'LSTM'
    ):

    if last_rnn_dense_layer:
        print("last_rnn_dense_layer is true means that output_hidden take rnn_units value")

    tracklets = Input((None, nb_feature), name="Trajectory Input")
    observation = Input(nb_feature, name="Observation Input")

    if RNN_layer == 'LSTM':
        rnn_net = build_LSTM(
            nb_feature, 
            nb_rrn_layer,
            siamese_unit, 
            use_dropout,
            dropout,
            use_activation,
            activation,
            last_rnn_dense_layer,
            name="Trajectory_feature_extractor_based_LSTM")
    elif RNN_layer == 'GRU':
        rnn_net = build_GRU(
            nb_feature,
            nb_rrn_layer, 
            siamese_unit, 
            use_dropout,
            dropout, 
            use_activation,
            activation,
            last_rnn_dense_layer,
            name="Trajectory_feature_extractor_based_GRU")
    else:
        raise ValueError


    dense_net = build_dense(
        nb_feature,
        nb_dense_layer, 
        siamese_unit, 
        use_dropout=use_dropout,
        dropout=dropout,
        use_activation=use_activation,
        activation=activation,
        name='Observation_feature_extractor'
        )

    vectA = rnn_net(tracklets)
    vectB = dense_net(observation)

    if batch_normalisation:
        vectA = BatchNormalization()(vectA)
        vectB = BatchNormalization()(vectB)

    if distance_metric is not None:

        distance = Lambda(distance_metric, name=distance_metric.__name__)([vectA, vectB])

        outputs = Dense(1, name="similarity", activation="sigmoid")(distance)

        return Model(inputs=[tracklets, observation], outputs=outputs, name="Trajectory_Observation_Associator")
    else:

        final_net = build_dense(
            2 * siamese_unit,
            nb_final_layer, 
            final_unit,
            reduce_unit_final,
            use_dropout,
            dropout,
            use_activation,
            activation,
            name="distance_metric_learning"
            )

        final_vect = final_net(tensorflow.concat([vectA, vectB], axis=1))
        if use_dropout:
            final_vect = Dropout(dropout)(final_vect)
        if use_activation:
            final_vect = Activation(activation)(final_vect)

        outputs = Dense(1, name='similarity', activation='sigmoid')(final_vect)

        return Model(inputs=[tracklets, observation], outputs=outputs, name="Trajectory_Observation_Associator")

if __name__=="__main__":
    feature_length = 4

    x1 = np.random.sample((10, feature_length))
    x2 = np.random.sample((10, feature_length))

    tracklets_obs_associator = similarity_measure_model(
        feature_length,
        distance_metric=euclidean_distance,
        nb_rrn_layer=2,
        nb_dense_layer=2,
        nb_final_layer=4,
        RNN_layer='GRU'
        )

    plot_model(tracklets_obs_associator, to_file='tracklets_obs_associator_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    print(x1)
    print()
    print(x2)
    print()
    print(tracklets_obs_associator([x1, x2]))