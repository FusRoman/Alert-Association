# Solar System Alert Association with Machine Learning

The goal of this previous work on alert association was to used machine learning and perform the associations with the known trajectories alerts provided by the cross match with the Minor Planet Center (MPC).

The first try was to used Graph Neural Network (GNN) to perform a link prediction task over an alert association graph.
The graph dataset are build in the graph directories. 

The file motlayer.py create the Message Passing Network (MPN) computation in order to aggregate more information on the graph. Motlayer extend the original Message Passing Network with a time notion between the alerts. 