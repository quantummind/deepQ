import numpy as np
from pyquil import Program
from pyquil.gates import *
from pyquil.api import get_qc

from rules import *
from circuit_families import *

# QFT(pi)
def qft_pi(dim):
    s = ''
    for i in range(dim):
        q1 = qc.qubits()[i]
        s += 'H ' + str(q1) + '\n'
        for j in range(i+1, dim):
            q2 = qc.qubits()[j]
            s += 'CZ ' + str(q2) + ' ' + str(q1) + '\n'
    return s

# dumb way to add commuting PRAGMAs:
    # look between CZ blocks (only single-qubit gates)
    # order all gates by qubit without changing order at the qubit level
    # layer by taking transpose
    # insert Is for each unused qubit / where qubit is incomplete
    # make each layer commute
    # make each CZ commute with a layer of identities on all other qubits
def commute_program(s):
    cz_blocks = s.split('CZ')
    all_qubits = qc.qubits()
    czs = []
    circuit = []
    for i in range(len(cz_blocks)):
        single_gates = cz_blocks[i].split('\n')[:-1]
        if i > 0:
            czs.append('CZ' + single_gates[0])
            single_gates = single_gates[1:]
            if single_gates[-1] == 'HALT':
                single_gates = single_gates[:-1]
            
            circuit.append([])
            cz_qubits = czs[-1].split(' ')[1:]
            place_at = int(cz_qubits[1])
            empty = int(cz_qubits[0])
#             place_at = np.searchsorted(all_qubits, int(cz_qubits[1]))
#             empty = np.searchsorted(all_qubits, int(cz_qubits[0]))
            for q in all_qubits:
                if q == place_at:
                    circuit[-1].append('CZ ' + cz_qubits[0])
                elif q == empty:
                    circuit[-1].append('')
                else:
                    circuit[-1].append('I')
        
        qubits = [int(g.split(' ')[1]) for g in single_gates]
        qubit_sorted_gates = [[] for q in all_qubits]
        for j in range(len(qubits)):
            qubit_sorted_gates[np.searchsorted(all_qubits, qubits[j])].append(single_gates[j])
        
        layered_gates = []
        num_layers = np.amax([len(row) for row in qubit_sorted_gates])
        for j in range(len(qubit_sorted_gates)):
            row = qubit_sorted_gates[j]
            row.extend([('I ' + str(all_qubits[j])) for k in range(num_layers - len(row))])
            layered_gates.append(row)
        new_block = ''
        for j in range(num_layers):
            circuit.append([])
            for k in range(np.amax(all_qubits)+1):
                if k in all_qubits:
                    circuit[-1].append(qubit_sorted_gates[np.searchsorted(all_qubits, k)][j].split(' ')[0])
    return np.array(circuit)

if __name__ == '__main__':
    qft_size = 5
    qc = get_qc("Aspen-4-5Q-C", as_qvm=True)
    qc.compiler.client.timeout = 60
    
    identities = pickle.load(open('output/identities-0-5-pi.pkl', 'rb'))
    # trim out qubit index in identities
    for k, v in identities.items():
        v2 = []
        for s in v:
            v2.append(s.replace(' 0\n', '\n').split('\n')[:-1])
        identities[k] = v2
    
    i = qft_size
        
    # insert layer of Hadamards before QFT, so true outcome is all 0s
    s = ''
    for j in range(i):
        s += 'H ' + str(qc.qubits()[j]) + '\n'
    s += qft_pi(i)
    p = Program(s)
    print('p', p)
    nq = qc.compiler.quil_to_native_quil(p)
    print('nq', str(nq))
    c = commute_program(str(nq))[:, :i]
    print('c', c)

    np.savez_compressed('output/qft-zeroed-' + str(qft_size) + '.npz', c)
        
#         family = [c]
#         for j in range(num_padded):
#             print('padded', j, 'out of', num_padded)
#             pc = pad_circuit(c, identities)
#             family.append(pc)
#         families.append(family)