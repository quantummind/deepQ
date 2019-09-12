import numpy as np
from multiprocessing import Pool

from pyquil import Program
from pyquil.gates import *
from pyquil.noise import add_decoherence_noise
from pyquil.api import get_qc

from circuit_families import *
from rules import *

def worker(nq):
    ex = qc.compiler.native_quil_to_executable(nq)
    ran = qc.run(ex)
    return ran

def run_circuit(c, nshots=1000, noise=True):
    p = circuit_to_program(qc, c, num_qubits=5)
    if noise:
        p = add_decoherence_noise(p, ro_fidelity=1)
    nq = qc.compiler.quil_to_native_quil(p)
    
    
    out = []

#     for i in range(nshots):
#         print(i)
#         out.append(worker(nq))
    
    with Pool(24) as pool:
        args = []
        for i in range(nshots):
            args.append((nq,))
        out = pool.starmap(worker, args)
        
    return out

def run_family(f, ind, noise=True):
    reads = []
    print('running family', ind)
#     np.save('output/simulations-random/readout-clean-' + str(ind) + '.npy', run_circuit(f[0], noise=False, nshots=10000))
    for i in range(len(f)):
        print(i, 'out of', len(f))
        reads.append(run_circuit(f[i], noise=noise))
        print(np.sum(reads[-1])/1000)
#         np.save('output/simulations-8x8/noise-' + str(ind) + '.npy', noise)
        np.save('output/simulations-random-5/readouts1000-noisy-' + str(ind) + '.npy', reads)

qc = get_qc('Aspen-4-5Q-C', as_qvm=True)
families = np.load('output/circuit-families-random-5.npz')['arr_0']

for i in range(len(families)):
    run_family(families[i], i)

# with Pool(20) as pool:
#     args = []
#     for i in range(len(families)):
#         args.append((families[i], i))
#     pool.starmap(run_family, args)