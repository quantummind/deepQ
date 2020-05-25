# Quantum circuit optimizer

A. Zlokapa and A. Gheorghiu, “A deep learning model for noise prediction on near-term quantum devices.” https://arxiv.org/abs/2005.10811

A. Zlokapa and A. Gheorghiu, “A deep learning approach to noise prediction and circuit optimization for near-term quantum devices.” *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis*, November 2019. 1st place, ACM SRC SC19. 2nd place, ACM SRC International.

## Abstract
We present an approach for a deep-learning compiler of quantum circuits, designed to reduce the output noise of circuits run on a specific device. We train a convolutional neural network on experimental data from a quantum device to learn a hardware-specific noise model. A compiler then uses the trained network as a noise predictor and inserts sequences of gates in circuits so as to minimize expected noise. We tested this approach on the IBM 5-qubit devices and observed a reduction in output noise of 12.3% (95% CI [11.5%, 13.0%]) compared to the circuits obtained by the Qiskit compiler. Moreover, the trained noise model is hardware-specific: applying a noise model trained on one device to another device yields a noise reduction of only 5.2% (95% CI [4.9%, 5.6%]). These results suggest that device-specific compilers using machine learning may yield higher fidelity operations and provide insights for the design of noise models. 

## Overview

* `main.ipynb`: Summary of results, comparing compilation with the deep learning noise model to the IBM Q compiler.
* `dd_analysis.py`: Demonstrates that learned sequences do not achieve dynamical decoupling to first order, despite reducing noise more than dynamical decoupling.

### Data

To run the code and notebooks, unzip the data available at https://caltech.box.com/s/jni52ra5qpq28f9tunaob9auwvisavb9 and add it directly to the main directory of the project.

### Circuit generation

* `generate_circuits.py`: Helper functions for making supremacy-style circuits with a layer of random single-qubit gates from {sqrt(X), sqrt(Y), sqrt(W)} followed by a CX gate between any two qubits. Generation of training set, `supremacy_all_5_unique/circuits.npy`.
* `make_test_circuits.py`: Generation of test circuits, `test_circuits_5.npy`.

### Deep learning

* `model.py`: Definition of neural network and circuit pair dataset structure.
* `train.py`: Train model on training set with 80% training set, 10% validation set (for early stopping), and 10% test set.
* `test_compiler.py`: Compile test circuits with maximum IBM Q compiler optimization and then pad the output of that with the deep learning model.
* `result_plotter.ipynb`: Evaluate model performance on training/validation/test set.

### Circuit Evaluation

* `run_circuits.py`: Run training set circuits, saved in `supremacy_all_5_unique/burlington_noise.npy` for Burlington and similarly for London.
* `run_test_compiled.py`: Run test set circuits, including free evolution and compiled, saved in `test_noise_5`.

## Citation

```
@misc{alex2020deep,
    title={A deep learning model for noise prediction on near-term quantum devices},
    author={Alexander Zlokapa and Alexandru Gheorghiu},
    year={2020},
    eprint={2005.10811},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```