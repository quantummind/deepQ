import numpy as np
from qiskit.circuit import QuantumCircuit
import re
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def u3(t, p, l):
    return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)], [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(l+p))*np.cos(t/2)]])

def dag(m):
    return m.conj().T

# get all gates belonging to a qubit
def get_gates(s, qubit):
    lines = re.findall(r'.+q\[' + str(qubit) + '\]', s)[:-2]
    for i in range(len(lines)):
        lines[i] = lines[i][:lines[i].index('q')-1]
    return lines

def parse_gate(g):
    if 'u3' in g:
        i1 = g.find(',')
        i2 = g.find(',', i1+1)
        n1 = g[g.find('(')+1:i1]
        n2 = g[i1+1:i2]
        n3 = g[i2+1:g.find(')')]
    elif 'u2' in g:
        i1 = g.find(',')
        n1 = np.pi/2
        n2 = g[g.find('(')+1:i1]
        n3 = g[i1+1:g.find(')')]
    elif 'u1' in g:
        n1 = 0
        n2 = 0
        n3 = g[g.find('(')+1:g.find(')')]
    else:
        print(g)
    return float(n1), float(n2), float(n3)

def get_dd_sequences(sf, sc):
    num_qubits = int(sf[sf.find('qreg') + 7:sf.find(']', sf.find('qreg'))])
    dd = []
    for q in range(num_qubits):
        free = get_gates(sf, q)
        compiled = get_gates(sc, q)
        if len(compiled) != len(free):
            offset = 0
            new_sequence = True
            current_sequence = []
            for i in range(len(compiled)):
                if i - offset >= len(free):
                    break
                if compiled[i] != free[i - offset]:
                    offset += 1
                    current_sequence.append(parse_gate(compiled[i]))
                else:
                    if len(current_sequence) != 0:
                        dd.append(current_sequence)
                        current_sequence = []
                    new_sequence = True
    return(dd)


def get_sequences(free, compiled):
    all_dd = []
    for i in range(len(free)):
        all_dd.extend(get_dd_sequences(free[i], compiled[i]))

        sequences = {}
    for dd in all_dd:
        if len(dd) not in sequences:
            sequences[len(dd)] = [dd]
        else:
            sequences[len(dd)].append(dd)
    out = {}
    for k in sorted(sequences.keys()):
        out[k] = np.array(sequences[k])
    return out


train = np.load('supremacy_all_5_unique/burlington_circuits.npy')
tf = np.tile(train[:, 0], 15)
tc = train[:, 1:].transpose().flatten()

bf = np.load('test_5_burlington_free.npy')
bc = np.load('test_5_burlington_compiled.npy')

lf = np.load('test_5_london_free.npy')
lc = np.load('test_5_london_compiled.npy')


b_sequences = get_sequences(bf, bc)
l_sequences = get_sequences(lf, lc)
train_sequences = get_sequences(tf, tc)

def process_seq(sequences):
    pauli_matrices = np.array((
                               ((0, 1), (1, 0)),
                               ((0, -1j), (1j, 0)),
                               ((1, 0), (0, -1))
                              ))

    avgs = []
    for d in sequences:
        pauli_avg = 0
        for H in pauli_matrices:
            total = np.zeros((2, 2), dtype=np.complex128)
            matrix = np.identity(2)
            for j in range(len(d)):
                matrix = np.matmul(matrix, u3(*d[len(d)-j-1]))
                total += np.matmul(dag(matrix), np.matmul(H, matrix))
            m = total/len(d)
            pauli_avg += np.linalg.eigh(m)[0][-1]
        pauli_avg /= len(pauli_matrices)
        avgs.append(pauli_avg)
    return np.array(avgs)

x = [3.14159265358979,0.0,3.14159265358979]
y = [3.14159265358979,1.57079632679490,1.57079632679490]
xyxy_test = [[x, y, x, y]]
train_random = process_seq(train_sequences[4])
b_random = process_seq(b_sequences[4])
l_random = process_seq(l_sequences[4])

print('True dynamical decoupling (DD) sequences, such as XYXY achieve first-order suppression of decoherence (ideal is 0):', process_seq(xyxy_test)[0])
print('Random sequences, DD suppresion of noise:', bs.bootstrap(train_random, stat_func=bs_stats.mean))
print('Burlington deep learning sequences, DD suppresion of noise:', bs.bootstrap(b_random, stat_func=bs_stats.mean))
print('London deep learning sequences, DD suppresion of noise:', bs.bootstrap(l_random, stat_func=bs_stats.mean))
print('All 95% confidence intervals overlap, and none of the values are near zero, showing first-order dynamical decoupling is not achieved by the deep learning sequences.')