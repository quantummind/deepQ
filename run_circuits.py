import numpy as np
from qiskit import *
import time
from qiskit.compiler import transpile
from qiskit.providers.jobstatus import JobStatus

IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_burlington')

group_size = 4
shots = 5000

all_circuits = np.load('supremacy_all_5_unique/circuits.npy')

circuits = []
for i in range(1000):
    a = all_circuits[i]
    for j in range(len(a)):
        s = a[j]
        circuits.append(QuantumCircuit.from_qasm_str(s))
    if (i+1) % group_size == 0:
        out = execute(circuits, backend, shots=shots, optimization_level=0)
        # check status regularly as workaround
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
        np.save('supremacy_all_5_unique_noise/' + str(i).zfill(5) + '.npy', f)
        print('completed', i)
        circuits = []