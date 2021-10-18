#import tensorflow
from os import lseek

import tensorflow
from src.model.LSTM.Quality_LSTM.model import quality_model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.metrics as kmetrics
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from sklearn.preprocessing import Normalizer


def trajectory_fit_normalizer(trajectory_dataset):
    normalizers = {}
    for i in range(trajectory_dataset.shape[0]):
        normalizers[i] = Normalizer()
        trajectory_dataset[i, :, :] = normalizers[i].fit_transform(trajectory_dataset[i, :, :]) 
    return normalizers

def trajectory_normalize_transform(trajectory_dataset, normalizers):
    for i in range(trajectory_dataset.shape[0]):
        trajectory_dataset[i, :, :] = normalizers[i].transform(trajectory_dataset[i, :, :]) 

def observation_normalize_transform(observation_dataset, normalizers):
    for i in range(len(observation_dataset)):
        observation_dataset[i] = normalizers[i].transform(observation_dataset[i].reshape((1, -1)))


if __name__=="__main__":

    model = quality_model(6)

    plot_model(model, to_file='quality_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    i = 5
    print("trajectory_length: {}".format(i))
    trajectory_dataset = np.load("../Siamese_LSTM/data/trajectory_dataset_length={}.npy".format(i), allow_pickle=True).astype('float64')
    observation_dataset = np.load("../Siamese_LSTM/data/observation_dataset_length={}.npy".format(i), allow_pickle=True).astype('float64')
    obs_shape = np.shape(observation_dataset)
    observation_dataset = observation_dataset.reshape((obs_shape[0], 1, obs_shape[1]))
    label_dataset = np.load("../Siamese_LSTM/data/label_dataset_length={}.npy".format(i), allow_pickle=True)

    feat = 6
    norm_traj = trajectory_fit_normalizer(trajectory_dataset)
    observation_normalize_transform(observation_dataset, norm_traj)

    print(model([trajectory_dataset, observation_dataset]))
    


    exit()
    obs_shape = np.shape(observation_dataset)

    print(np.shape(trajectory_dataset))
    print(np.shape(observation_dataset))

    training_trajectory = trajectory_dataset[::2]
    validation_trajectory = trajectory_dataset[1::2]

    training_observation = observation_dataset[::2]
    validation_observation = observation_dataset[1::2]

    training_label = label_dataset[::2]
    validation_label = label_dataset[1::2]

    print(model([training_trajectory, training_observation]))
    

    exit()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            kmetrics.BinaryAccuracy(),
            kmetrics.Precision(),
            kmetrics.Recall(),
            kmetrics.TruePositives(),
            kmetrics.TrueNegatives(),
            kmetrics.FalsePositives(),
            kmetrics.FalseNegatives(),
        ]
    )

    history = model.fit(
        x=[training_trajectory, training_observation],
        y=training_label,
        validation_data=((validation_trajectory, validation_observation), validation_label),
        epochs=50,
    )


    plot_history(history, path="history_training.png")
    plt.close()