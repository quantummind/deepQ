import pickle
import numpy as np
from numpy import pi
from random import randrange
from scipy.stats import powerlaw

from pyquil import Program
from pyquil.gates import *
from pyquil.api import get_qc
from pyquil.api import WavefunctionSimulator

from rules import *
from multiprocessing import Pool

def random_program_all_gates(qc, n_gates):
    qubits = np.array(qc.qubits()).astype(str)

    gate_set = ['X', 'Y', 'Z', 'H', 'S', 'T', 'CZ', 'CNOT', 'CCNOT', 'SWAP', 'CSWAP', 'ISWAP']
    gates_2q = ['CZ', 'CNOT', 'SWAP', 'ISWAP']
    gates_3q = ['CCNOT', 'CSWAP']
    s = ''
    for i in range(n_gates):
        g = gate_set[randrange(len(gate_set))]
        remaining_qubits = qubits
        if g in gates_2q:
            i1 = randrange(len(remaining_qubits))
            q1 = remaining_qubits[i1]
            remaining_qubits = np.delete(remaining_qubits, i1)
            i2 = randrange(len(remaining_qubits))
            q2 = remaining_qubits[i2]
            s += g + ' ' + q1 + ' ' + q2 + '\n'
        elif g in gates_3q:
            i1 = randrange(len(remaining_qubits))
            q1 = remaining_qubits[i1]
            remaining_qubits = np.delete(remaining_qubits, i1)
            i2 = randrange(len(remaining_qubits))
            q2 = remaining_qubits[i2]
            remaining_qubits = np.delete(remaining_qubits, i2)
            i3 = randrange(len(remaining_qubits))
            q3 = remaining_qubits[i3]
            s += g + ' ' + q1 + ' ' + q2 + ' ' + q3 + '\n'
        else:
            s += g + ' ' + remaining_qubits[randrange(len(remaining_qubits))] + '\n'
    p = Program(s)
    
    ro = p.declare('ro', 'BIT', len(qubits))
    for i, q in enumerate(qc.qubits()):
        p += MEASURE(q, ro[i])
    
    return p

# circ is a list with each row of size # qubits
# if an element is empty (''), it means it's the first element of a CZ gate
def circuit_to_str(qc, circ, pragmas=True, preserve=True):
    qubits = qc.qubits()
    s = ''
    if preserve:
        s += 'PRAGMA PRESERVE_BLOCK\n'
    for layer in circ:
        if pragmas:
            s += 'PRAGMA COMMUTING_BLOCKS\n'
        for i in range(len(layer)):
            c = layer[i]
            if c != '':
                if pragmas:
                    s += 'PRAGMA BLOCK\n'
                s += c + ' ' + str(qubits[i]) + '\n'
                if pragmas:
                    s += 'PRAGMA END_BLOCK\n'
        if pragmas:
            s += 'PRAGMA END_COMMUTING_BLOCKS\n'
    if preserve:
        s += 'PRAGMA END_PRESERVE_BLOCK\n'
    return s

def circuit_to_program(qc, circ, num_qubits=-1, pragmas=True, preserve=True, measure=True):
    p = Program(circuit_to_str(qc, circ, pragmas=pragmas, preserve=preserve))
    
    if num_qubits == -1:
        num_qubits = len(qc.qubits())
    
    if measure:
        ro = p.declare('ro', 'BIT', num_qubits)
        for i, q in enumerate(qc.qubits()[:num_qubits]):
            p += MEASURE(q, ro[i])
        
    return p

def random_circuit(qc, dims, id_prob=0.5, symmetrize=False):
    # RZ argument random float
    
    # generate qubit by qubit then pad identities and insert '' whenever a CZ is made
    
    circ = np.zeros(dims).astype(str)
    
    depth = dims[0]
    if symmetrize:
        depth = depth // 2
    qubit_range = dims[1]
    
    qubits = np.array(qc.qubits()[:qubit_range])
    
#     qubits = qubits.astype(str)
#     gate_set = create_gate_set()
    gates_2q = ['CZ']
    edges_raw = qc.get_isa().edges
    edges = []
    for e in edges_raw:
        in_target = True
        for s in e.targets:
            if s not in qubits:
                in_target = False
                break
        if in_target:
            edges.append([s for s in e.targets])
    edges = np.array(edges)
    
    c = []
    rx_args = ['pi/2', 'pi', '-pi/2']
    
    for i in qubits:
        row = np.array(['' for j in range(depth)])
        
        # first put in identities in streaks so ~half the gates are identity
        run_lengths = powerlaw.rvs(0.03, loc = 0, scale = depth - 1, size = int(depth/10 * id_prob)) + 1
        run_lengths = np.round(run_lengths).astype(int)
        
        # put identities in random locations, fusing all overlapping identities
        entries = np.random.randint(0, depth, len(run_lengths))
        for r in range(len(run_lengths)):
            row[entries[r]:entries[r] + run_lengths[r]] = 'I'
        
        # create a list of gates that can't be simplified where there are empty gaps in the row
        # since CZ has a second qubit, set probabilities of [RX, RZ, CZ] to [0.2, 0.2, 0.1]
        nonidentity = np.where(row != 'I')[0]
        row = row.tolist()
        for j in nonidentity:
            if j == 0:
                prev_gate = 'I'
            else:
                prev_gate = row[j-1]

            if prev_gate == 'I':
                new_gate = np.random.choice(['RX', 'RZ', 'CZ'], p=[0.4, 0.4, 0.2])
            elif 'RX' in prev_gate:
                new_gate = np.random.choice(['RZ', 'CZ'], p=[0.67, 0.33])
            elif 'RZ' in prev_gate:
                new_gate = np.random.choice(['RX', 'CZ'], p=[0.67, 0.33])
            elif 'CZ' in prev_gate:
                new_gate = np.random.choice(['RX', 'RZ'], p=[0.5, 0.5])

            if new_gate == 'RX':
                new_gate += '(' + np.random.choice(rx_args) + ')'
            elif new_gate == 'RZ':
                new_gate += '(' + str(np.random.uniform(-np.pi, np.pi)) + ')'
            elif new_gate == 'CZ':
                possible_edges = edges[np.where(edges[:, 1] == i)]
                if len(possible_edges) > 0:
                    e = possible_edges[randrange(len(possible_edges))]
                    new_gate += ' ' + str(e[0])
                else:
                    new_gate = 'I'
            
            row[j] = new_gate
        c.append(row)
    
    c = np.array(c).T
    
    # blank out gates and add identities according to placement of CZ gates
    for i in range(len(c)):
        # only allow one CZ per layer
        # make everything else identity or '' (other side of CZ)
        layer = c[i]
        cz_inds = []
        for j in range(len(layer)):
            if 'CZ' in layer[j]:
                cz_inds.append(j)
        if len(cz_inds) > 0:
            cz_ind = np.random.choice(cz_inds)
            cz = layer[cz_ind]
            other_gate = int(cz.split(' ')[1])
            other_ind = np.searchsorted(qubits, other_gate)
            c[i, :] = 'I'
            c[i][cz_ind] = cz
            c[i][other_ind] = ''
    
    return c

# return U U^\dagger
def dagger_gates(gates):
    daggered = []
    for i in range(len(gates)):
        if gates[i] in ['', 'CZ', 'I']:
            daggered.append(gates[i])
        else:
            negative = '(-1*'.join(gates[i].split('('))
            daggered.append(negative)
    return np.array(daggered)

def symmetrize_circuit(c, dims):
    c_sym = np.empty((dims[0], c.shape[1])).astype(str)
    
    # insert column of identities if needs to be odd depth
    if dims[0] % 2 == 1:
        c_sym[c.shape[0], :] = 'I'
    c_sym[:c.shape[0]] = c
    for i in range(len(c)):
        c_sym[len(c_sym) - i - 1] = dagger_gates(c[i])
    
    return c_sym

def shuffle_pad_circuit(circ_orig, identities):
    # move identities around with non-identities
    # randomly re-arrange gates in a row (between CZ or '')
    circ = np.copy(circ_orig)
    tcirc = circ.T
    for i in range(len(tcirc)):
        merged = []
        for j in range(len(tcirc[i])):
            if 'CZ' in tcirc[i][j] or tcirc[i][j] == '':
                merged.append(j)
        merged = np.array(merged)
        for j in range(len(merged) - 1):
            segment = tcirc[i][merged[j]+1:merged[j+1]]
            nonidentities = segment[np.where(segment != 'I')]
            inds = np.arange(len(segment))
            inds = np.sort(np.random.choice(inds, size=len(nonidentities), replace=False))
            new_seg = np.array(['I' for k in range(len(segment))]).astype(circ.dtype)
            new_seg[inds] = nonidentities
            tcirc[i][merged[j]+1:merged[j+1]] = new_seg
    return pad_circuit(tcirc.T, identities)

def pad_circuit(circ_orig, identities):
    circ = np.copy(circ_orig)
    tcirc = circ.T
    max_len = np.amax(list(identities.keys())) + 1
    for i in range(len(tcirc)):
        id_inds = np.where(tcirc[i] == 'I')[0]
        run_inds = np.split(id_inds, np.where(np.diff(id_inds) != 1)[0]+1)
        
        for r in run_inds:
            if len(r) > 1:
                # if a run is short enough to be found in the identities dictionary, substitute it
                # if a run is too long, combine shorter identities
                lengths = []
                while len(r) - np.sum(lengths) > max_len:
                    lengths.append(1 + randrange(1, max_len))
                if np.sum(lengths) - len(r) != 0:
                    lengths.append(len(r) - np.sum(lengths))
                all_ids = []
                for l in lengths:
                    potential_ids = identities[l-1]
                    all_ids.extend(potential_ids[randrange(len(potential_ids))])
                tcirc[i][r] = all_ids
    return tcirc.T

# TODO multithread
def generate_circuit_families(qc, dims, num_unique, num_padded, symmetrize=False):
    identities = pickle.load(open('output/identities-0-5-pi.pkl', 'rb'))
    # trim out qubit index in identities
    for k, v in identities.items():
        v2 = []
        for s in v:
            v2.append(s.replace(' 0\n', '\n').split('\n')[:-1])
        identities[k] = v2
    
    families = []
    for i in range(num_unique):
        print(i, 'out of', num_unique)
        c = random_circuit(qc, dims, symmetrize=symmetrize)
        if symmetrize:
            c = symmetrize_circuit(c, dims)
        family = [c]
        for j in range(num_padded):
            pc = shuffle_pad_circuit(c, identities)
#             zero = wf_simulate_circuit(qc, pc)['00000000']
#             if zero < 0.95:
#                 print('FAILED PADDING')
#             print(zero)
            family.append(pc)
        families.append(family)
    return np.array(families)

def wf_simulate_program(p):
    wf_sim = WavefunctionSimulator()
    wf = wf_sim.wavefunction(p)
    clean = wf.get_outcome_probs()
    return clean

def wf_simulate_circuit(qc, c):
    p = circuit_to_program(qc, c, measure=False)
    return wf_simulate_program(p)

def run_program(qc, p):
    nq = qc.compiler.quil_to_native_quil(p)
    ep = qc.compiler.native_quil_to_executable(nq)
    return qc.run(ep)

if __name__ == '__main__':
    qc = get_qc("Aspen-4-5Q-C", as_qvm=True)
    qft_circuit = np.load('output/qft-zeroed-5.npz')['arr_0']
    
    num_unique = 500
    num_padded = 20
    
#     c = random_circuit(qc, qft_circuit.shape)
#     print(c)
#     np.save('output/test-circuit.npy', c)
    
    families = generate_circuit_families(qc, qft_circuit.shape, num_unique, num_padded, symmetrize=True)
    np.savez_compressed('output/circuit-families-random.npz', families)

#     depth = 4
#     num_unique = 500
#     num_padded = 20
#     families = generate_circuit_families(qc, depth, num_unique, num_padded, num_qubits=8, symmetrize=True)
#     np.savez_compressed('circuit-families-8x8-sym.npz', families)