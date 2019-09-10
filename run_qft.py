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

def run_circuit(c, num_qubits, nshots=1000):
    p = circuit_to_program(qc, c, num_qubits=num_qubits)
#     p_noisy = add_decoherence_noise(p, ro_fidelity=1)
    nq = qc.compiler.quil_to_native_quil(p)
#     ep = qc.compiler.native_quil_to_executable(nq)
    
    out = []
    with Pool(24) as pool:
        args = []
        for i in range(nshots):
            args.append((nq,))
        out = pool.starmap(worker, args)
        
    return np.array(out)

def run_family(f, ind):
    reads = []
    print('running family', ind)
    for i in range(len(f)):
        print(i, 'out of', len(f))
        reads.append(run_circuit(f[i], len(f[i][0])))
        print(np.average(reads[-1], axis=0))
        np.save('qft/readouts-' + str(ind) + '.npy', reads)

# qc = get_qc('Aspen-4-16Q-A', as_qvm=True)
qc = get_qc('Aspen-4-14Q-C', as_qvm=True)
families = np.load('circuit-families-qft-new.npz', allow_pickle=True)['arr_0']

for i in range(len(families)):
    run_family(families[i], i)

# with Pool(20) as pool:
#     args = []
#     for i in range(len(families)):
#         args.append((families[i], i))
#     pool.starmap(run_family, args)