# Training

This catalog contains scripts used to train NN models for the likelihood regression.

## File list

+ **train.py** -- The main script that is used to train NN models and export them to ONNX files.
+ **NNmodel.py** -- Implementation of a MLP-based network for likelihood regression.
+ **BNNmodel.py** -- Implementation of a Bayesian MLP-based network for likelihood regression **[EXPERIMENTAL]**.
+ **optimize.py** -- Implementation of the hyperparameter optimisation procedure.
+ **losses.py** -- Some useful functions for the loss calculation.
+ **time_check.py** -- A little script that can be used to estimate the execution time of an NN model.