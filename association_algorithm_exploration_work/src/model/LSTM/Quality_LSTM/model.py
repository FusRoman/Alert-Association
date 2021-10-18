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
import tensorflow.keras.metrics as kmetrics
import numpy as np
from plot_keras_history import plot_history
import matplotlib.pyplot as plt


def quality_model(input_shape):

    trajectory_inputs = Input((None, input_shape))
    observation_inputs = Input((None, input_shape))

    _, state_h, state_c = LSTM(1, return_state=True)(trajectory_inputs)

    quality_embedding = LSTM(1)(observation_inputs, initial_state=[state_h, state_c])

    quality = tensorflow.keras.activations.sigmoid(quality_embedding)
    return Model(inputs=[trajectory_inputs, observation_inputs], outputs = quality)

if __name__=="__main__":
    model = quality_model(4)

    traj_test = np.random.sample((5, 5, 4))
    obs_test = np.random.sample((5, 1, 4))

    print(model([traj_test, obs_test]))

    t = np.array([[[1], [2], [3], [4]],
        [[2], [4], [8], [16]],
        [[1], [3], [5], [7]],
        [[2], [4], [6], [8]]])


    r1 = np.array([
        [[5]],
        [[32]],
        [[9]],
        [[10]]
    ])

    r2 = np.array([
        [[12]],
        [[7]],
        [[13]],
        [[5]]
    ])

    t_train = np.concatenate([t, t])
    o_train = np.concatenate([r1, r2])
    label = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    print(t_train)
    print(o_train)
    print(label)

    model = quality_model(1)
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
        x=[t_train, o_train],
        y=label,
        epochs=10000
    )

    plot_history(history, path="history_training.png")
    plt.close()

    exit()


    input = Input((4, 1))
    input2 = Input((1, 1))
    o, h, c = LSTM(1, return_state=True)(input)

    o2 = LSTM(1)(input2, initial_state=[h, c])

    res_m = tensorflow.keras.activations.sigmoid(o2)

    m = Model([input, input2], outputs=res_m)

    res_m = m([t, r1])

    print(res_m)


    res_m = m([t, r2])
    print("-----")

    print(res_m)
