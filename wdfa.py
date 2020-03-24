from collections import defaultdict, namedtuple, Counter
from functools import reduce
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image


Transition = namedtuple("Transition", ['weight', 'node_index'])

class State:

    def __init__(self, reaching=0, terminating=0, transitions=dict(), name=None, number=None):

        self.transitions = transitions
        self.terminating = terminating
        self.reaching = reaching
        
        self.name = name
        self.number = number
    
        
    @property
    def is_leaf(self):
        
        return self.transitions == dict()
    
    def __repr__(self):
        
        return f"{self.number}[{self.reaching},{self.terminating}]"
        
class WDFA:
    
    def __init__(self, initial_state=None):
        
        self.initial_state = initial_state if initial_state is not None else State(number=0)
        self.states = {self.initial_state.number: self.initial_state}
        self.number_of_states = 1
        self.alphabet = set()
        self.canonical_projection = [0]

            
    def PPTA(self, sample_words, counts=False, normalise=True):
        
        if not counts:
            terminating = Counter(sample_words)
        
        reaching = reduce(
            lambda a,b:a+b, 
            [
                scale_counter(
                    counter=Counter(get_prefixes(w)),
                    scalar=terminating[w]
                    
                )
                for w in terminating
            ]
        )
        
        alphabet = get_minumum_alphabet(sample_words)
        prefixes = lexicographic_sort(reaching)
        ordinals=inverse_dict(prefixes)

        states = {
            i:State(
                reaching=reaching[prefixes[i]],
                terminating=terminating[prefixes[i]],
                name=prefixes[i],
                number=i,
                transitions={
                    a:Transition(
                        weight=reaching[prefixes[i]+a], 
                        node_index=ordinals[prefixes[i]+a]
                    )

                    for a in alphabet
                    if (
                        (prefixes[i]=='' and a in prefixes)
                        or
                        (prefixes[i]+a in prefixes)
                    )
                }
            )
            for i in range(len(prefixes))
        }
        
        self.alphabet = alphabet
        self.states = states
        self.initial_state = states[0]
        self.canonical_projection = list(range(len(prefixes)))
    
    def normalise(self):
        
        for node_index in self.states:
            
            normalise_state_transitions(self.states[node_index])
    
    def class_of(self, node_index):
        
        if node_index < len(self.canonical_projection):
            
            return self.canonical_projection[node_index]
        
        else:
            
            None
    
    def get_state_of_node(self, node_index):
        
        state_index = self.class_of(node_index)
        state = self.states[state_index]
        
        return state
    
    def generate_transition_matrices(self):
        
        number_of_states = max(self.states)
        
        row = defaultdict(lambda :[])
        col = defaultdict(lambda :[])
        data = defaultdict(lambda :[])
        
        # default value is a matrix in csr format filled with zeros
        M = defaultdict(lambda : csr_matrix(([],([],[])),shape=(number_of_states, number_of_states)))
     
        for idx, current_state in self.states.items():
            
            if not current_state.is_leaf:
                for input_symbol, (weight, node_index) in current_state.transitions.items():

                    next_state = self.get_state_of_node(node_index)
                        
                    row[input_symbol].append(current_state.number)
                    col[input_symbol].append(next_state.number)
                    data[input_symbol].append(weight)
        
        
        symbols = list(data.keys())
        
        for symbol in symbols:
            M[symbol] = csr_matrix(
                (data[symbol], (row[symbol], col[symbol])),
                shape=(number_of_states, number_of_states)
            )
        
        return M
     
    def networkx_graph(self):
        
        N = nx.MultiDiGraph()
        
        for source_state_index in self.states:
            
            source_state = self.states[source_state_index]
            if not source_state.is_leaf:
                for input_symbol, (weight, target_node_index) in source_state.transitions.items():
                    
                    target_state_index = self.class_of(target_node_index)
                    
                    N.add_edge(
                        source_state_index, 
                        target_state_index, 
                        input_symbol=input_symbol,
                        weight=weight,
                        label=f'{input_symbol}[{weight}]'
                    )
                     
        
        labels = {
            node:str(self.states[node])
            for node in N.nodes()
        }
        
        nx.set_node_attributes(N, {k: {'label': labels[k]} for k in self.states})
        
        return N

        

        
    def plot(self):
        
        N = self.networkx_graph()
        
        D = nx.drawing.nx_agraph.to_agraph(N)

        D.layout('dot')
    
        D.draw('buffer.png')
        
        return Image('buffer.png')
            

def normalise_state_transitions(state):
    
    if state.is_leaf:
        
        state.terminating = 1.
        
    else:
        
        summ = state.terminating + sum(weight for input_symbol, (weight, state) in state.transitions.items())

        state.terminating = state.terminating / summ

        for input_symbol, (weight, next_state) in state.transitions.items():

            probability = weight/summ
            state.transitions[input_symbol] = (probability, next_state)
        
        
def hoeffding_different(f1, n1, f2, n2, alpha):

    return abs(f1/n1 - f2/n2) > np.sqrt(0.5 * np.log(2/alpha))*(1/np.sqrt(n1) + 1/np.sqrt(n2))

def hoeffding_compatible(wdfa, state1, state2, alpha):
    
    if state1 == state2:
        
        return True
    
    f1, f2 = state1.terminating, state2.terminating
    n1, n2 = state1.reaching, state2.reaching
    
    if hoeffding_different(f1, n1, f2, n2, alpha):
        
        return False
    
    for symbol in wdfa.alphabet:
        
        transitions1 = state1.transitions
        transitions2 = state2.transitions
        
        if symbol in transitions1 and symbol in transitions2:
            
            f1, f2 = transitions1[symbol].weight, transitions2[symbol].weight

            if hoeffding_different(f1, n1, f2, n2, alpha):
                return False
            
            target_node_index1, target_node_index2 = transitions1[symbol].node_index, transitions2[symbol].node_index
            target_state1, target_state2 = wdfa.get_state_of_node(target_node_index1), wdfa.get_state_of_node(target_node_index2)

            if target_state1 == state1 and target_state2 == state2:
                continue
            
            if not hoeffding_compatible(wdfa, target_state1, target_state2, alpha):
                
                return False

        elif symbol in transitions1:
            
            f1 = transitions1[symbol].weight
            
            if hoeffding_different(f1, n1, 0, n2, alpha):

                return False
            
        elif symbol in transitions2:
            
            f2 = transitions2[symbol].weight
            
            if hoeffding_different(0, n1, f2, n2, alpha):

                return False
        
    return True
    
    
def merge(wdfa, state_index1, state_index2):
    
    state1, state2 = wdfa.states[state_index1], wdfa.states[state_index2]
        
    transitions1 = state1.transitions
    transitions2 = state2.transitions
    
    transitions = dict()
    for symbol in wdfa.alphabet:
        
        if symbol in transitions1 and symbol in transitions2:
            
            transition1, transition2 = transitions1[symbol], transitions2[symbol]
            
            next_node_index1, next_node_index2 = transition1.node_index, transition2.node_index 
            
            next_state_index1, next_state_index2 = wdfa.class_of(next_node_index1), wdfa.class_of(next_node_index2)
            
            if next_state_index1 != next_state_index2:
                
                merge(wdfa, next_state_index1, next_state_index2)
                
                state1, state2 = wdfa.states[state_index1], wdfa.states[state_index2]

                transitions1 = state1.transitions
                transitions2 = state2.transitions
            

            weight1, weight2 = transition1.weight, transition2.weight
                
            transitions[symbol] = Transition(weight1 + weight2, next_state_index1)
            
        elif symbol in transitions1:
                
            transitions[symbol] = transitions1[symbol]
                
        elif symbol in transitions2:
                
            transitions[symbol] = transitions2[symbol]
            
    (s,t) = (state_index1, state_index2) if state_index1 <= state_index2 else (state_index2, state_index1)
    for node_index in range(len(wdfa.canonical_projection)):

        if wdfa.canonical_projection[node_index] == t:
            wdfa.canonical_projection[node_index] = s
                
    merged_state = State(
        number=s,
        transitions=transitions,
        terminating=state1.terminating+state2.terminating,
        reaching=state1.reaching+state2.reaching
    )
        
    wdfa.states[s] = merged_state
    del wdfa.states[t]

    
def alergia(sample_words, counts=False, alpha=0.5):
    #import pdb; pdb.set_trace()

    M = WDFA()
    
    M.PPTA(
        sample_words=sample_words,
        counts=counts,
        normalise=False
    )
    
    state_indices = sorted(M.states.keys())
    
    for j in state_indices[1:]:
        if j in M.states:
            lower_state_indices = (i for i in state_indices if i<=j and i in M.states)

            i = next(lower_state_indices)
            state_i = M.states[i]
            state_j = M.states[j]

            while not hoeffding_compatible(M, state_i, state_j, alpha=alpha):
                i = next(lower_state_indices)
                state_i = M.states[i]


            if i < j:
                merge(M, i, j)

    return M


def lexicographic_sort(strings):
    
    strings_copy = list(strings)
    strings_copy.sort(key=lambda item: (len(item), item))
    
    return strings_copy

def get_minumum_alphabet(strings):
    
    return set(
        a 
        for string in strings
        for a in string
    )

def get_prefixes(string):
    
    return [string[:i] for i in range(len(string)+1)]

def inverse_dict(lst):
    
    return dict(map(
        lambda pair:(pair[1],pair[0]),
        enumerate(lst)
    ))

def scale_counter(counter, scalar):
    
    return Counter({
        key:scalar*count
        for key, count in counter.items()
    })
