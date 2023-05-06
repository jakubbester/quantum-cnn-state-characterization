Final project for CPSC 452/552 Spring 2023.

### Team members:
Jakub Bester, Nikhil Harle, Frank Li, Ioannis Panitsas, Connor Totilas

## Introduction
We implement a **Q**uantum **C**onvolutional **N**eural **N**etwork (QCNN) and variations using the `torchquantum` package, following the work of [Cong *et al.* (2019)](https://www.nature.com/articles/s41567-019-0648-8/) to distinguish whether an input state is a simulation of a topological Floquet Majorana mode, using code from the work of [Harle *et al.* (2023)](https://www.nature.com/articles/s41467-023-37725-0). The QCNN works as a binary classifier, where we are given an input Majorana circuit, perform quantum convolutional and pooling layers, input measurements in a fully connected MLP layer, and output a binary prediction of the classification of the topological Floquet Majorana mode.

As described in the code summary below, we have several models in `models.py`, generate the Majorana mode datasets with `state_prep.py`, and run these models in the `qcnn_training.ipynb` notebook with the `train_funcs.py`. In `models.py`, we have the classical neural network, the pure quantum network, the base hybrid quantum neural network, the hybrid quantum neural network with different unitaries, and the hybrid quantum neural network with multiple feature maps. These models can all be run in parallel so that they can be compared in performance. We found that the we were able to achieve 95% classification accuracy with our best model. 

## Code Summary
You can train any of the QCNN models/variations in the `qcnn_training.ipynb` notebook, where the training process is nicely explained.
The source code is in the `source` folder, and consists of four files:
1. `models.py`
Contains all of the QCNN architectures, the classical benchmarking, and variations on the base architecture.
2. `state_prep.py`
Contains code to generate Majorana modes.
3. `train_funcs.py`
Contains training and testing implementations/logic.
4. `utils.py`
Contains various utility functions.