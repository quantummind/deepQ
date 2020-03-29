import numpy as np
from qiskit import *
from qiskit.compiler import transpile
from generate_circuits import pad_circuit

IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_5_yorktown')
# backend = provider.get_backend('ibmq_ourense')

bucket_size = 50
shots = 5000

identities = np.load('test_free.npy')
compiled = np.load('test_compiled.npy')

def run_type(data, imin, imax):
    circuits = []
    for s in data[imin:imax]:
        circuits.append(QuantumCircuit.from_qasm_str(s))
    print('number of circuits', len(circuits))
    out = execute(circuits, backend, shots=shots, optimization_level=0)
    res = out.result()
    f = []
    for qc in circuits:
        counts = res.get_counts(qc)
        ones = 0
        for k, v in counts.items():
            ones += v * k.count('1')
        f.append(ones/shots)
    return f

def run(imin, imax, i, d='test_noise_5_only_yorktown/'):
    id_freqs = run_type(identities, imin, imax)
    co_freqs = run_type(compiled, imin, imax)
    
    np.save(d + 'run2__identity_' + str(i).zfill(5) + '.npy', id_freqs)
    np.save(d + 'run2_compiled_' + str(i).zfill(5) + '.npy', co_freqs)
    
for i in range(14):
    run(i*bucket_size, (i+1)*bucket_size, i)

# for i in range(len(compiled)//bucket_size):
#     run(i*bucket_size, (i+1)*bucket_size, i)
# if len(compiled) % bucket_size != 0:
#     div = len(compiled)//bucket_size
#     run(bucket_size*div, len(compiled), div)