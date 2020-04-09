from collections import Counter
from functools import reduce
import numpy as np
from automatools.wdfa import WDFA, State, Transition


def PPTA(sample_words, counts=False, normalise=True):

    ppta = WDFA()
    
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

    ppta.alphabet = alphabet
    ppta.states = states
    ppta.initial_state = states[0]
    ppta.canonical_projection = list(range(len(prefixes)))
    
    if normalise:
        ppta.normalise()
    
    return ppta


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
  
    M = PPTA(
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


