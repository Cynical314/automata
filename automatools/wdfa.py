from collections import defaultdict, namedtuple, Counter
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
        self.alphabet = set()
        self.canonical_projection = [0]

    @property
    def number_of_nodes(self):
        
        return len(self.canonical_projection)
    
    @property
    def number_of_states(self):
        
        return len(self.states)
        
    @property
    def is_quotiented(self):
        
        return self.number_of_nodes == self.number_of_states
    
    
    def delta(self, state, input_symbol):
        
        transition = state.transitions[input_symbol]
        
        next_state = self.get_state_of_node(transitions.node_index)
        weight = transition.weight
        
        return next_state, weight
    
    def get_weight_of(self, word):
        
        if word == list() or word == str():
            
            return self.initial_state.terminating
        
        else:
            
            first_symbol = word[0]
        
        I, TransitionMats, T = self.generate_transition_matrices()
        
        Product = TransitionMats[first_symbol]
        for symbol in word[1:]:
            Product *= TransitionMats[symbol]
        
        return (I*Product*T).toarray()[0][0]
    
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
        
        if self.is_quotiented:
            
            number_of_states = self.number_of_states
            
        else:
            
            number_of_states = max(self.states)+1

        row = defaultdict(lambda :[])
        col = defaultdict(lambda :[])
        data = defaultdict(lambda :[])

        # default value is a matrix in csr format filled with zeros
        M = defaultdict(lambda : csr_matrix(([],([],[])), shape=(number_of_states, number_of_states)))

        for idx, current_state in self.states.items():

            if not current_state.is_leaf:
                for input_symbol, (weight, node_index) in current_state.transitions.items():

                    next_state = self.class_of(node_index)

                    row[input_symbol].append(idx)
                    col[input_symbol].append(next_state)
                    data[input_symbol].append(weight)


        symbols = list(data.keys())

        for symbol in symbols:
            M[symbol] = csr_matrix(
                (data[symbol], (row[symbol], col[symbol])),
                shape=(number_of_states, number_of_states)
            )

        I = csr_matrix(([1],([0],[0])), shape=(1, number_of_states))

        T = csr_matrix(
            (
                [self.states[i].terminating for i in sorted(self.states)],
                (sorted(self.states), self.number_of_states*[0])
            ),
            shape=(number_of_states, 1)
        )

        return I, M, T

    
    
    
    def quotient_nodes(self):
        
        if not self.is_quotiented:
        
            state_indices = sorted(self.states)

            for idx, state in self.states.items():
                for input_symbol, transition in state.transitions.items():

                    next_node_index = transition.node_index
                    next_state_index = self.class_of(next_node_index)
                    ordinal_of_next_state = state_indices.index(next_state_index)

                    state.transitions[input_symbol] = Transition(
                        weight=transition.weight,
                        node_index=ordinal_of_next_state
                    )

                ordinal_of_state = state_indices.index(idx)
                state.number = ordinal_of_state


            self.states = {

                state_indices.index(i): state

                for i, state in self.states.items()
            }

            self.canonical_projection = list(range(len(self.states)))

        
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
                        label=f'{input_symbol}[{weight if type(weight)==int else weight:.2f}]'
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
            state.transitions[input_symbol] = Transition(
                weight=probability, 
                node_index=next_state
            )
        
        
