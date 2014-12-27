arrhythmia-nn
=============
##### Cardiac Dysrhythmia Detection with GPU-Accelerated Neural Networks

### Poster & Paper
If you want an overview, you should read the [poster](http://albert.cm/dl/arrhythmia_poster.pdf). The [paper](http://albert.cm/dl/arrhythmia_paper.pdf) is more technical about algorithms and implementation details. The poster is slightly outdated so view the paper for the most current results.

### Models
We use [scikit-learn](http://scikit-learn.org/stable/) for training and testing of our learning algorithms. Located in [`./python/`](./python) are python scripts which train each model. The models are listed below:

* Multiclass Logistic Regression
* Support Vector Machine (one-vs-all)
* Random Forest
* Neural Network

### Data
We use the [Arrhythmia Data Set](https://archive.ics.uci.edu/ml/datasets/Arrhythmia) which is part of the UCI Machine Learning Repository. Our imputed dataset is located at
[`./data/data_clean_imputed.csv`](./data/data_clean_imputed.csv) and contains the clean data. As mentioned in the paper, [`./data/pca.csv`](./data/pca.csv) contains the principal components of the clean dataset. Matlab code to impute the original dataset is found in  [`./matlab/impute.m`](./matlab/imputem) and is compatible with Octave.