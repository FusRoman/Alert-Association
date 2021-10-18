# Solar System Object Tracking with Graph Association

Using graph to perform alert association in order to discover new asteroid was our first try. 

Graph are construct over all night of one month and each association of one night are connected to all alerts in other night. Spektral library are used to build the graphs and the loader for the training.

Nodes contains 'static' features of solar system object (sso) candidates.
This features are :

- ra, dec : right ascension, declinaison (2d coordinates to locate alerts in the sky)
- dcmag : apparent magnitude computed by fink function
- fid : filter used during exposition by telescope (affects the magnitude)
- nid : night identification of the shooting night
- jd : julian date at the start of the exposition

Edges contains 'dynamic' features of sso candidates.
This features are :

- angular separation : distance between the alerts projected on a sphere, computed by astropy
- magnitude difference : magnitude difference between the alerts
- nid difference
- jd difference

Angular separation and magnitude difference are normalized by jd difference

- motgraph.py extend the graph definition of Spektral to add the time notion and edge information
- motgraphdataset.py build a dataset of motgraph from the alert database
- motloader.py is a loader design to simplified the training process and iteration over the dataset of motgraph. 