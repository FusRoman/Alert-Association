# Alert-Association

This project use alerts from ZTF tagged like asteroids by Fink. The goal of this project is to associate alerts issued of the same object.

## Model

Alert association use a graph neural network to learn association from known objects from Fink. The task is a link prediction on the alerts graph. The model must be able to learn the link which associates alerts of the same object. 

## Technology

Graph Neural Network and Message Passing Network use Spektral library which is an extension of Keras/TensorFlow.

Alerts comes from Fink and graph is constructed with Spark.

