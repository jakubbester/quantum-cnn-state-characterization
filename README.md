Final project for CPSC 452/552 Spring 2023.

### Team members:
Jakub Bester, Nikhil Harle, Frank Li, Ioannis Panitsas, Connor Totilas

## Project Experimental Directions:
- comparison to basic neural network
- increasing the number of filters / trying different unitaries
- increasing the depth of the conv net
- changing loss function and optimizer
- Trying on another topological state
- Noise Reduction with Encoding/Decoding + Res Net: https://arxiv.org/abs/2012.07772

## Project timeline:
Now-April 15th: get barebones version running.
- Make dataset (https://github.com/IBM/observation-majorana)
- Import QCNN code into Github (https://www.tensorflow.org/quantum/tutorials/qcnn)
- Running and troubleshooting

April 15th-April 22nd: go off in different directions (see above). 
- 2 people try different topological state
- 3 people try adjusting structure

April 22nd: Reconvene and start putting together paper!
- Introduction
- Problem Setting/Motivation
- QCNN approach
- What we do differently
- Results

## List of possible directions:
1. Try for different topological states (e.g. https://arxiv.org/abs/2210.17548)
2. Optimization for fault-tolerant operations on QEC code spaces
3. Experiment with other loss functions, adjustments to QCNN structure, etc.
- Implement parallel feature maps
- Implement backpropogation
- Implement skip connections


## Concrete project directions:
1. Implementing multiple feature maps (shared weights) - Frank
  - Making all QCNN quantum gates non trainable as a baseline comparison (purely classical NN)
  - Do multiple feature maps with trainable unitaries
  - Make sure we're implementing shared weights :)
  
2. Try for different (topological) state - Connor
  - Find Qiskit circuit that prepares the state and add code to Github in the form of a function like QCNN.prepare_majorana()
  - Ideas: AKLT, cluster state with excitation, whatever we want!
  
3. Play around with the QCNN structure - Jakub, Ioannis
  - Try to find different trainable unitary gates
  - Try making all gates in same conv/pool layer the same
  
Side things:  
1. Running trained model on actual hardware (?!) - Nikhil

2. Exploring QCAE (basically just adding circuit again but backwards lol) - Nikhil
