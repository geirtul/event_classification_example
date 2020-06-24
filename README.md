# Classifying events in a nuclear physics experiment - an example
Standalone repository for classification example on simulated data that can be used in ML course.

# Content
## Notebooks
├── notebooks\
│   ├── convolutional.ipynb\
│   ├── data_import_exploration.ipynb\
│   ├── data_import_exploration_traces.ipynb\
│   ├── dense_neural_network.ipynb\
│   ├── helper_functions.py\
│   ├── logistic.ipynb\
└── README.md\

### data_import_exploration
This notebook contains a walkthrough from the first import of the data, formatting the data,
initial analysis and exploration, and finally storing the data in a desired format.

### logistic
Scikit-Learn example on using logistic regression to classify the data

### dense
Keras example on using a fully-connected neural network to classify the data

### convolutional
Keras example on using a convolutional neural network to classify the data

## Data 
├── data\
│   ├── CeBr10k_1.txt\
│   ├── CeBr200k_Mix.txt.gz\

The repository includes one file with 10000 events (CeBr10k_1), and one file with 200k events (CeBr200k_Mix.txt.gz).
The latter is compressed to comply with githubs max file size for regular repositories.


# TODO:
* Add example on model saving
* Go through texts and see if it fits well with the rest of the material
