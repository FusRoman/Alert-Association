# Solar System Alert Association with Machine Learning

The goal of this previous work on alert association was to used machine learning and perform the associations with the known trajectories alerts provided by the cross match with the Minor Planet Center (MPC).

Our first try was to used Graph Neural Network (GNN) to perform a link prediction task over an alert association graph based on this paper :
[Learning a Neural Solver for Multiple Object Tracking](https://arxiv.org/abs/1912.07515).

The graph dataset is built in the graph directory. 

The file motlayer.py perform the Message Passing Network (MPN) computation in order to aggregate more information on the graph. Motlayer extend the original Message Passing Network with a time notion between the alerts. 

The file motmodel.py build the neural network model which used motlayer.

The file trajectory.py build trajectory based on the result produce by motmodel.

## Limit

This solution to perform alert association has been abandonned mainly for two reason.

- Firstly, the graph construction is too memory expensive and need to be optimized. Graphs are build over one month of alerts. Every month, they have <img src="https://render.githubusercontent.com/render/math?math=%2410%5E%7B5%7D%24"> MPC alerts and <img src="https://render.githubusercontent.com/render/math?math=%2410%5E%7B2%7D%24"> Solar System Object (SSO) Candidates provided by Fink with the ZTF Alerts Stream. The graph need to be almost complete (removal of edges is done with physical constraints on asteroids) so the graph size in terms of edges is bounded by <img src="https://render.githubusercontent.com/render/math?math=%24O(n%5E%7B2%7D)%24"> where <img src="https://render.githubusercontent.com/render/math?math=%24n%20%3D%2010%5E%7B5%7D%24">. We used the number of MPC alerts to measure performance since we expect this order of magnitude for SSO Candidates in LSST.

- Secondly, the training of the MOTModel doesn't converge to an optimal solution of the alert association problem. The problems can occur in the MOTLayer implementation. The index computation for past edges and futur edges can be bugged. The four MLP defined for MOTLayer can be used incorrectly. In addition, the hyperparameters doesn't have been totally optimized. Finally, it is possible that the Graph Neural Network is not the right solution to solve this problem. 