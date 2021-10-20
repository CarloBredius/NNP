# Visual exploration of the generalization of Neural Network Projections
A Master thesis by Carlo Bredius.

---
## Setup
One requires to have at least the following versions:
- Java 8
- Octave 4
- Anaconda for python 3.x
- Python packages as follows
'''pip install scipy scikit-learn scikit-image MulticoreTSNE tensorflow keras keras-applications wget umap-learn joblib pandas numpy matplotlib PyQtGraph

Then one can clone the repository and run the tool.

---
## How to use
1. To load in datasets, one runs the file 0_get_data.py. When a dataset is needed that does not have a function for it yet, write a new function that start with 'process_', and it'll automatically be run when the python file is executed.

2. To train a new Neural Network Projection model, one runs the file 1_train_nnp.py. In code, one has to load the dataset and projeciton technique by uncommenting them. One can also train a multitude of NNP models in the for-loop. In the file nnproj.py one can change the architecture/configuration of the neural network.

3. To run the tool, one runs the file 2_scatter_tool.py. In the function loadData(), one needs to determine what NNP model is loaded and how many samples are shown in the tool. 