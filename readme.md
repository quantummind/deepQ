# Quantum circuit optimizer


A. Zlokapa and A. Gheorghiu, “A deep learning approach to noise prediction and circuit optimization for near-term quantum devices.” *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis*, November 2019.

The project focuses on intelligently compiling random circuits with dynamical decoupling to minimize noise according to a trained ResNet noise model. **For an overview of results, see** `main.ipynb`.

## Overview

* `main.ipynb`: Summary of results, comparing deep learning dynamical decoupling compiler to the IBM Q compiler.

### Circuit generation

* `generate_circuits.py`: Helper functions for making supremacy-style circuits with a layer of random single-qubit gates from {sqrt(X), sqrt(Y), sqrt(Z)} followed by a CX gate between any two qubits. Generation of training set, `supremacy_all_5_unique/burlington_circuits.npy`.
* `make_test_circuits.py`: Generation of test circuits, `test_circuits_5.npy`.

### Deep learning

* `model.py`: Definition of neural network and circuit pair dataset structure.
* `train.py`: Train model on training set with 80% training set, 10% validation set (for early stopping), and 10% test set.
* `test_compiler.py`: Compile test circuits with maximum IBM Q compiler optimization (saved in `test_free.npy`) and then pad the output of that with the deep learning model (saved in `test_compiled.npy`).

### Circuit Evaluation

* `run_circuits.py`: Run training set circuits, saved in `supremacy_all_5_unique/burlington_noise.npy`.
* `run_test_compiled.py`: Run test set circuits, including free evolution and compiled. Save in `test_noise_5_only_*` depending on selected device.