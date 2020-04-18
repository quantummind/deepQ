import numpy as np
from qiskit import *
from qiskit.compiler import transpile
from multiprocessing import Pool
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.extensions import *
from qiskit.extensions.unitary import UnitaryGate
from scipy.linalg import schur

def power(gate, exponent):
    """Creates a unitary gate as `gate^exponent`.

    Args:
        exponent (float): Gate^exponent

    Returns:
        UnitaryGate: To which `to_matrix` is self.to_matrix^exponent.

    Raises:
        CircuitError: If Gate is not unitary
    """
    from qiskit.extensions.unitary import UnitaryGate  # pylint: disable=cyclic-import
    # Should be diagonalized because it's a unitary.
    decomposition, unitary = schur(gate.to_matrix(), output='complex')
    # Raise the diagonal entries to the specified power
    decomposition_power = list()

    decomposition_diagonal = decomposition.diagonal()
    # assert off-diagonal are 0
    if not np.allclose(np.diag(decomposition_diagonal), decomposition):
        raise CircuitError('The matrix is not diagonal')

    for element in decomposition_diagonal:
        decomposition_power.append(pow(element, exponent))
    # Then reconstruct the resulting gate.
    unitary_power = unitary @ np.diag(decomposition_power) @ unitary.conj().T
    return UnitaryGate(unitary_power)

def WGate():
    return UnitaryGate((XGate().to_matrix() + YGate().to_matrix())/np.sqrt(2))

def rand_cx(n):
    q1 = np.random.randint(n)
    q2 = np.delete(np.arange(n), q1)
    q2 = np.random.choice(q2, size=1)[0]
    return 'cx q[' + str(q1) + '],q[' + str(q2) + '];\n', q1, q2

def supremacy_circuit(bend=None, m=20, n=5):
    sqrtx = 'u3(1.57079632679490,-1.57079632679490,1.57079632679490)'
    sqrty = 'u3(1.57079632679490,0,0)'
    sqrtw = 'u3(1.57079632679490,-0.785398163397448,0.785398163397449)'
    gates = [sqrtx, sqrty, sqrtw]
    
    qasm_base = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[n];
creg c[n];"""
    
    last_gates = -np.ones(n, dtype=np.int64)
    s = qasm_base.replace('[n]', '[' + str(n) + ']')
    for i in range(m):
        # single-qubit gates
        for j in range(n):
            choices = np.arange(len(gates))
            if last_gates[j] != -1:
                choices = np.delete(choices, last_gates[j])
            g = np.random.choice(choices, size=1)[0]
            last_gates[j] = g
            s += gates[g] + ' q[' + str(j) + '];\n'
        
        # two-qubit gate
        g, q1, q2 = rand_cx(n)
        s += g
        last_gates[q1] = -1
        last_gates[q2]

    first_half = QuantumCircuit.from_qasm_str(s)
    qc = transpile(first_half + first_half.inverse(), backend=bend, optimization_level=0)
    s = qc.qasm()
    for i in range(n):
        s += 'measure q[' + str(i) + '] -> c[' + str(i) + '];\n'
    return s

def generate_gate(ind, bend, inc_fraction):
    if ind < inc_fraction**3:
        i = ind // inc_fraction**2
        j = (ind // inc_fraction) % inc_fraction
        k = ind % inc_fraction
        return 'u3(' + str(i/inc_fraction*np.pi) + ', ' + str(j/inc_fraction*np.pi) + ', ' + str(k/inc_fraction*np.pi) + ')'
    else:
        return None

# return a random num_gates length identity
def id_m(num_gates, bend, inc_fraction=6):
    num_possibilities = inc_fraction**3
    qasm_base = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];

"""
    try_again = True
    
    while try_again:
        try_again = False
        s = qasm_base
        identity = []
        for i in range(num_gates - 1):
            g = generate_gate(np.random.randint(num_possibilities), bend, inc_fraction)
            identity.append(g)
            s += g + ' q[0];\n'

        first_gates = QuantumCircuit.from_qasm_str(s)

        last_gate = transpile(first_gates.inverse(), backend=bend, optimization_level=3)
        if len(last_gate.data) == 0:
            g = generate_gate(0, bend, inc_fraction)
        elif last_gate.size() == 1:
            g = last_gate.qasm().split('\n')[-2][:-6]
        else:
            try_again = True
        
        identity.append(g)

    return identity

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def pad_circuit(qasm_code, bend, n=5):
    s3 = qasm_code

    # create array to determine where gaps are, assuming latest-as-possible scheduling
    trimmed = s3[s3.find('creg'):s3.find('barrier')].splitlines()[1:]
    trimmed.reverse()
    circuit_arr = np.zeros((n, len(trimmed)), dtype=np.int64)
    for ind in range(len(trimmed)):
        line = trimmed[ind]
        offset = 0
        qubits = []
        for i in range(2):
            q1_start = line.find('q', offset)
            if q1_start == -1:
                break
            q1_end = line.find(']', q1_start)
            qubit = line[q1_start+2:q1_end]
            qubits.append(int(qubit))
            offset = q1_end
        col_num = []
        for q in qubits:
            if len(np.nonzero(circuit_arr[q])[0]) != 0:
                col_num.append(np.max(np.nonzero(circuit_arr[q])))
        if len(col_num) == 0:
            col_num = 0
        else:
            col_num = np.max(col_num) + 1
        for q in qubits:
            circuit_arr[q][col_num] = 1 + ind

    col_num = []
    for q in range(n):
        if len(np.nonzero(circuit_arr[q])[0]) != 0:
            col_num.append(np.max(np.nonzero(circuit_arr[q])))
    if len(col_num) == 0:
        return None
    col_num = np.max(col_num) + 1
    circuit_arr = circuit_arr[:, :col_num]

    # add padding
    inc_fraction = 6
    padding = []
    padding_arr = np.zeros(circuit_arr.shape, dtype=np.int64)
    for i in range(n):
        zeros = zero_runs(circuit_arr[i])
        identities = []
        for z in zeros:
            if z[1] != circuit_arr.shape[1] and z[1] - z[0] > 1:
                identity = id_m(z[1] - z[0], bend)
                initial_length = len(padding)
                for j in range(len(identity)):
                    d = identity[j] + ' q[' + str(i) + '];'
                    padding.append(d)
                    padding_arr[i][z[0] + j] = initial_length + len(identity) - j
    
    if len(padding) == 0:
        return None
    
    padded_circuit = []
    used_lines = []
    for i in range(circuit_arr.shape[1]):
        for j in range(circuit_arr.shape[0]):
            l = circuit_arr[j][i]
            if l == 0:
                if padding_arr[j][i] != 0:
                    padded_circuit.append(padding[padding_arr[j][i] - 1])
            elif l not in used_lines:
                used_lines.append(l)
                padded_circuit.append(trimmed[l - 1])
    padded_circuit.reverse()
    pc = s3[:s3.find('\n', s3.find('creg'))] + '\n' + '\n'.join(padded_circuit) + '\n' + s3[s3.find('barrier'):]
    
    return pc


def make_family(ind, length=16, n=5, directory='./supremacy_all_5_unique/'):
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

            family.append(c)
            for i in range(length-1):
                family.append(pad_circuit(c, backend, n=n))
                if family[1] is None:
                    try_again = True
                    break
            if not try_again:
                np.save(directory + str(ind).zfill(5) + '.npy', family)
    
if __name__ == '__main__':
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    backend = provider.get_backend('ibmq_burlington')
    
    num_circuits = 1000
    with Pool(28) as p:
        p.map(make_family, np.arange(num_circuits))