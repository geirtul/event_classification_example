# Classifying events in a nuclear physics experiment - an example
Standalone repository for classification example on simulated data that can be used in ML course.

# Content
## /notebooks

### data_import_exploration
This notebook contains a walkthrough from the first import of the data, formatting the data,
initial analysis and exploration, and finally storing the data in a desired format.

### data_import_exploration_traces
A starting point for working with the traces datafile.

### logistic
Scikit-Learn example on using logistic regression to classify the data. Presents the metrics we use
to assess the performance of a binary classifier.

### dense
Keras example on using a fully-connected neural network to classify the data

### convolutional
Keras example on using a convolutional neural network to classify the data

### helper_functions.py
Collection of functions developed throughout the notebooks that are useful to store
for easier re-use.

## /data 
* CeBr10k_1.txt
  * file with 10000 events
* CeBr200k_Mix.txt.gz
  * file with 200k events, compressed to comply with githubs max file size for regular repositories.
* training_pm_nosat_150k.dat.gz
  * file with 150k traces, compressed.


# TODO:
* Add example on model saving
* data exploration
    * correlations
    * energy distributions
    * position distributions (including separation distances)
