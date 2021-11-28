# Visual exploration of the generalization of Neural Network Projections
A Master thesis by Carlo Bredius.

![trail_map](/trail_map.png)
An interactive tool that allows the user to perturb the data to explore how it changes the projection inferred through Neural Network Projection. Due to instant feedback in the scatter plot interface, one easily gets a good feeling for how a perturbation changes a projection. Five simple perturbations are implemented, together with four different visualizations and their configurations to explore the changed projections from different angles.

---
## Setup
One requires to have at least the following versions:
- Java 8
- Octave 4
- Anaconda for python 3.x
- Python packages as follows
'''pip install scipy scikit-learn scikit-image MulticoreTSNE tensorflow keras keras-applications wget umap-learn joblib pandas numpy matplotlib PyQtGraph

Then, one can clone the repository and run the tool.

---
## How to use
1. To load in datasets, one runs the file 0_get_data.py. When a dataset is needed that does not have a function for it yet, write a new function that start with 'process_', and it'll automatically be run when the python file is executed.

2. To train a new Neural Network Projection model, one runs the file 1_train_nnp.py. In code, one has to load the dataset and projection technique by uncommenting them. One can also train a multitude of NNP models in the for-loop. In the file nnproj.py one can change the architecture/configuration of the neural network.

3. To run the tool, one runs the file 2_scatter_tool.py. In the function loadData(), one needs to determine what NNP model is loaded and how many samples are shown in the tool. 