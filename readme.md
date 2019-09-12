# Quantum circuit optimizer

In preparation for:

A. Zlokapa and A. Gheorghiu, “A deep learning approach to noise prediction and circuit optimization for near-term quantum devices.” *IEEE/ACM International Conference on High Performance Computing, Networking, Storage and Analysis*, November 2019.

The project focuses on intelligently compiling random circuits with dynamical decoupling to minimize noise according to a trained ResNet noise model.

## Important Requirements

Aside from a typical environment (running Python 3 with packages such as `numpy` and `scipy`), the code requires `pyquil` (tested with version 2.72) and the Rigetti `qvm` (1.7.2) and `quilc` (1.7.2). Additionally, PyTorch 1.1.0 is used for the deep learning.

## Running

### Rigetti environment

To generate random quantum circuits, the Rigetti compiler and virtual machine must be running in server mode. We recommend using `screen` or `tmux` to run `qvm -S` and `quilc -S` in separate consoles. For more information on setting up the Rigetti environment, see the [docs](http://docs.rigetti.com/en/stable/start.html).

### Circuit generation

To create a list of circuit patterns equivalent to the identity gate (used for dynamical decoupling), run `python3 rules.py`. This will write a `.pkl` file in the `output` directory.

To create a quantum Fourier transform circuit (by default with 5 qubits), run `python3 qft.py`. This circuit is used to determine the size of the random circuits in the neural network's training dataset. Note that the QFT circuit begins with a layer of Hadamard gates and uses CZ gates for the controlled phase, ensuring that the ideal output is a bitstring of zeros.

To create random non-QFT circuits with dynamical decoupling padding, run `python3 circuit_families.py`. This will generate families of equivalent circuits that have different padding over the identity gates. These circuits will have the same dimensions as the QFT circuit generated above, and they are generated as `UU^\dagger` to guarantee an ideal output of zeros.

To simulate the random circuits with noise and measure the distance from the ideal zero bitstring, run `python3 noise_dataset.py`. (Warning: this will take very long. Also, a directory may need to be manually created to successfully evaluate `run_family`.)

### Deep learning

To train a ResNet on the circuit dataset, run `CUDA_VISIBLE_DEVICES=0 python3 train.py`. Training with multiple GPUs is supported, but in practice the ResNet model is relatively small and thus a single GPU is typically sufficient.

To analyze the predicted noise compared to the target noise, run `CUDA_VISIBLE_DEVICES=0 python3 analyze_results.py`.

### Monte Carlo optimization

Coming soon!