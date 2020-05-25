import numpy as np
from qiskit import *
from qiskit.compiler import transpile
from generate_circuits import pad_circuit
from qiskit.providers.jobstatus import JobStatus
import time

computer = 'burlington'

IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_' + computer)

bucket_size = 50
shots = 5000

identities = np.load('test_5_burlington_free.npy')
compiled = np.load('test_5_burlington_compiled.npy')

def run_type(data, imin, imax):
    circuits = []
    for s in data[imin:imax]:
        circuits.append(QuantumCircuit.from_qasm_str(s))
    print('number of circuits', len(circuits))
    out = execute(circuits, backend, shots=shots, optimization_level=0)
    while out.status() is not JobStatus.DONE:
        time.sleep(10)
    res = out.result()
    f = []
    for qc in circuits:
        counts = res.get_counts(qc)
        ones = 0
        for k, v in counts.items():
            ones += v * k.count('1')
        f.append(ones/shots)
    return f

def run(imin, imax, i, d='test_noise_5_burlington/'):
    id_freqs = run_type(identities, imin, imax)
    co_freqs = run_type(compiled, imin, imax)
    
    np.save(d + computer + 'identity_' + str(i).zfill(5) + '.npy', id_freqs)
    np.save(d + computer + 'compiled_' + str(i).zfill(5) + '.npy', co_freqs)
    
for i in range(10):
    run(i*bucket_size, (i+1)*bucket_size, i)