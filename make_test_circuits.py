import numpy as np
from qiskit import *
import glob
from generate_circuits import supremacy_circuit, pad_circuit
from qiskit.transpiler.exceptions import TranspilerError

if __name__ == '__main__':
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    backend = provider.get_backend('ibmq_burlington')
    
    n_circuits = 1000
    n = 5
    circuits = []
    for i in range(n_circuits):
        print(i)
        
        try_again = True
        while try_again:
            try_again = False
            family = []
            s = supremacy_circuit(n=n, m=5)

            # compile to optimization_level=3
            qc = QuantumCircuit.from_qasm_str(s)
            try:
                qc_t = transpile(qc, backend=backend, optimization_level=3)
            except TranspilerError:
                try_again = True

            if not try_again:
                c = qc_t.qasm()

                p = pad_circuit(c, backend, n=n)
                if p is None:
                    try_again = True
        
        circuits.append(c)
    circuits = np.array(circuits)
    np.save('test_circuits_5.npy', circuits)