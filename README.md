arrhythmia-nn
=============
##### Cardiac Dysrhythmia Detection with GPU-Accelerated Neural Networks

### Poster & Paper
If you want an overview, you should read our [poster](http://albert.cm/dl/arrhythmia_poster.pdf). The [paper](http://albert.cm/dl/arrhythmia_paper.pdf) is more technical about algorithms and implementation details. The poster is slightly outdated so view the paper for the most current results.

### Models
We use [scikit-learn](http://scikit-learn.org/stable/) for training and testing everything except neural networks. Located in [`./python/`](./python) are python scripts which train each model. The models are listed below:

* Multiclass Logistic Regression
* Support Vector Machine (one-vs-all)
* Random Forest

### Neural Network Implementation
Neural networks are implemented in MATLAB with the [Neural Network Toolbox](http://www.mathworks.com/products/neural-network/). These files are found in the [`./matlab/`](./matlab) folder. Each file is explained below:

[`./matlab/nn_main.m`](./matlab/nn_main.m) - Iteratively trains several neural networks by varying several hyperparameters, training set size, and train/test ratios

[`./matlab/nn_single_iter.m`](./matlab/nn_single_iter.m) - Trains a single neural network using the specified parameters

[`./matlab/make_rse_plots.m`](./matlab/make_rse_plots.m) and [`./matlab/make_accuracy_plots.m`](./matlab/make_accuracy_plots.m) - Take input files containing results of the neural network tuning stage, generate, and format plots.

To train the network on the GPU, you must have the Mathwork's [Parallel Computing toolbox](http://www.mathworks.com/products/parallel-computing/).

### Data
We use the [Arrhythmia Data Set](https://archive.ics.uci.edu/ml/datasets/Arrhythmia) which is part of the UCI Machine Learning Repository. Our imputed dataset is located at
[`./data/data_clean_imputed.csv`](./data/data_clean_imputed.csv) and contains the clean data. As mentioned in the paper, [`./data/pca.csv`](./data/pca.csv) contains the principal components of the clean dataset. Matlab code to impute the original dataset is found in  [`./matlab/impute.m`](./matlab/imputem) and is compatible with Octave.