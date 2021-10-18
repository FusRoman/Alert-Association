import numpy as np
import tensorflow
from tensorflow.python.keras import callbacks
from src.model.LSTM.Siamese_LSTM.model import similarity_measure_model
import tensorflow.keras.metrics as kmetrics
import tensorflow.keras.optimizers as koptimizers
from tensorflow.keras.callbacks import EarlyStopping
from plot_keras_history import plot_history
from plot_keras_history import chain_histories
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from src.model.LSTM.Siamese_LSTM.model import euclidean_distance
from sklearn.preprocessing import Normalizer


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tensorflow.math.square(y_pred)
        margin_square = tensorflow.math.square(tensorflow.math.maximum(margin - (y_pred), 0))
        return tensorflow.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def trajectory_fit_normalizer(trajectory_dataset):
    normalizers = {}
    for i in range(trajectory_dataset.shape[0]):
        normalizers[i] = Normalizer()
        trajectory_dataset[i, :, :] = normalizers[i].fit_transform(trajectory_dataset[i, :, :]) 
    return normalizers

def trajectory_transform(trajectory_dataset, normalizers):
    for i in range(trajectory_dataset.shape[0]):
        trajectory_dataset[i, :, :] = normalizers[i].transform(trajectory_dataset[i, :, :]) 

def observation_transform(observation_dataset, normalizers):
    for i in range(len(observation_dataset)):
        observation_dataset[i] = normalizers[i].transform(observation_dataset[i].reshape((1, -1)))

if __name__=="__main__":

    i = 19
    print("trajectory_length: {}".format(i))
    trajectory_dataset = np.load("data/trajectory_dataset_length={}.npy".format(i), allow_pickle=True).astype('float64')
    observation_dataset = np.load("data/observation_dataset_length={}.npy".format(i), allow_pickle=True).astype('float64')
    label_dataset = np.load("data/label_dataset_length={}.npy".format(i), allow_pickle=True)

    print(trajectory_dataset)
    print()

    # preprocessing
    trajectories_normalizers = trajectory_fit_normalizer(trajectory_dataset)
    print(len(trajectories_normalizers))
    observation_transform(observation_dataset, trajectories_normalizers)

    print(trajectory_dataset)
    print()
    print(observation_dataset)

    training_trajectory = trajectory_dataset[::2]
    validation_trajectory = trajectory_dataset[1::2]

    training_observation = observation_dataset[::2]
    validation_observation = observation_dataset[1::2]

    training_label = label_dataset[::2]
    validation_label = label_dataset[1::2]

    tracklets_observation_associator = similarity_measure_model(
        np.shape(trajectory_dataset)[2],
        distance_metric=euclidean_distance,
        nb_rrn_layer=3,
        nb_dense_layer=3,
        nb_final_layer=2,
        siamese_unit=12,
        final_unit=12,
        reduce_unit_final=True,
        use_dropout=False,
        dropout=0.3,
        use_activation=True,
        activation='relu',
        last_rnn_dense_layer=False,
        batch_normalisation = False,
        RNN_layer = 'LSTM'
        )

    plot_model(tracklets_observation_associator, to_file='tracklets_obs_associator_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    exit()
    polynomial_learning_rate_decay = koptimizers.schedules.PolynomialDecay(0.1, 10000, 0.01)
    adamax_optimizer = koptimizers.Adam(learning_rate=polynomial_learning_rate_decay)

    recall_early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    tracklets_observation_associator.compile(
        optimizer=adamax_optimizer,
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

    history = tracklets_observation_associator.fit(
        x=[training_trajectory, training_observation],
        y=training_label,
        validation_data=((validation_trajectory, validation_observation), validation_label),
        epochs=50,
        callbacks=[recall_early_stopping]
    )


    plot_history(history, path="history_training.png")
    plt.close()