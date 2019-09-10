# create 1-qubit patterns equivalent to the identity for Rigetti
# work in Rigetti basis: RX, RZ

import pickle
import numpy as np
from numpy import pi
from itertools import combinations_with_replacement, permutations

from pyquil import Program
from pyquil.gates import *
from pyquil.unitary_tools import program_unitary


# check if 2 matrices are equal up to a phase
def phase_equal(m1, m2):
    return np.linalg.matrix_rank(np.column_stack((m1.flatten(), m2.flatten()))) == 1

# check if 2 circuits are equal up to a phase
def equal(p, q):
    c1 = program_unitary(p, 1)
    c2 = program_unitary(q, 1)
    
    return phase_equal(c1, c2)

def is_identity(gate_string):
    return phase_equal(program_unitary(Program(gate_string), 1), identity_matrix)

def create_gate_set(cz=True, qubit_0=False):
#     z_split = 4
#     x_gates = np.array([pi/2, pi, 3*pi/2]).astype('str')
#     z_gates = (pi/z_split * (1 + np.arange(2*z_split - 1))).astype('str')
    
    x_gates = ['pi/2', 'pi', '-pi/2']
    z_gates = ['pi/4', 'pi/2', '3*pi/4', 'pi', '5*pi/4', '3*pi/2', '7*pi/4']
    
    gate_set = ['I']
    
    if cz:
        gate_set = ['CZ']
    
    for g in x_gates:
        gate_set.append('RX(' + g + ')')
    for g in z_gates:
        gate_set.append('RZ(' + g + ')')
    
    if qubit_0:
        for i in range(len(gate_set)):
            gate_set[i] += ' 0\n'
    
    return gate_set

def find_identities(n_gates):
    # return dictionary {# of gates: [all string replacements], ...}
    
    gate_set = create_gate_set(cz=False, qubit_0=True)
    combos = list(set(combinations_with_replacement(gate_set, n_gates)))
    all_identities = []
    for c in combos:
#         # for now, remove all combinations that are just RZ
#         merged = ''.join(c)
#         if 'I' not in merged and 'X' not in merged:
#             continue
        perm = list(set(permutations(c)))
        circuits = np.array([''.join(p) for p in perm])
        identities = np.array([is_identity(p) for p in circuits])
        all_identities.extend(circuits[identities].tolist())
    return all_identities

if __name__ == '__main__':
    identity_matrix = program_unitary(Program(I(0)), 1)
    
    d = {}
    for i in range(6):
        ids = find_identities(i+1)
        d[i] = ids
        print('for circuits of length', i, 'generated', len(ids), 'identity patterns')
        pickle.dump(d, open('identities-0-5-pi.pkl', 'wb+'))