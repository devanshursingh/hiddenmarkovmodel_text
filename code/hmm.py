#!/usr/bin/env python3

import numpy as np
from softmax import softmax
from scipy.special import logsumexp

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

class HiddenMarkovModel():

    def __init__(self, document: str, num_states: int):

        self.num_states = num_states        # number of hidden states
        self.V = 27                # number of character types

        # Create dictionary that maps alphabet & ' ' to index 0-27 for emission matrix
        alphabet = list(string.ascii_lowercase).append(' ')
        vocab_index = [num for num in range(27)]
        self.vocab = {alphabet[i] : vocab_index[i] for i in range(len(vocab_index))}

        self.init_params()     # create and initialize params
        #self.alt_init_params()

        # Choose whether initial hidden state will be s1 or s2 with equal probability
        self.initial_state = random.choice([state for state in range(self.num_states)])

        with open(document, 'r') as f:
            self.doc = f.read().split('')


    def init_params(self):
        # Initializes an emission matrix of size 2 by 27 for the two states and 27 characters
        # Turns this matrix into log probabilities
        emission_row1 = np.concatenate((np.repeat(0.0370, 13), np.repeat(0.0371, 13), np.array([0.0367])))
        emission_row2 = np.concatenate((np.repeat(0.0371, 13), np.repeat(0.0370, 13), np.array([0.0367])))
        self.emission = np.log(np.concatenate((emission_row1, emission_row2), axis=0))

        # Initializes a transition matrix of size 2x2 for the two states
        # Turns this matrix into log probabilities
        transition = [[0.49, 0.51],
                      [0.51, 0.49]]
        self.transition = np.log(np.array(transition))
    
    def alt_init_params(self):
        pass

    def forward(self):
        self.alphas = np.full((self.num_states, len(self.doc)), fill_value=-inf)

        # For each character
        for char_id in np.arange(len(self.doc)):
            # First time step
            if char_id == 0:
                # Set intial hidden state alpha probs to equal
                # This assumes an intial start state with no emissions
                #self.alphas[:,char_id] = np.repeat(1 / self.num_states, self.num_states)

                # Alternatively, set intial state alpha to 1 (0 in log)
                # All others 0 (-inf in log)
                self.alphas[char_id, self.initial_state] = 0.
                continue
            
            # Vocab id is this character's index in the emission matrix
            vocab_id = self.vocab.get(self.doc[char_id])

            # Adding log probabilities of the previous alphas, the transitions and the emission
            # Should be a self.num_states by self.num_states matrix
            # array[0,1] = log(alpha0) + log(p('e'|s0)) + log(p(s1|s0))
            # From s0 to s1 ^^^
            # Need to expand dims to make broadcasting work
            a = np.expand_dims(self.alphas[:,char_id-1], axis=1)
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            intmed = a + self.transition + e

            # Sum all of the probs in the columns, which means logsumexping all of the log probs
            self.alphas[:,char_id] = logsumexp(intmed, axis=0)

        # Final time step, single end state where all alphas are summed
        char_id = -1 # last char
        vocab_id = self.vocab.get(self.doc[char_id])

        # Final sum of all alpha probs multiplied by the emission probs of the last char
        # Carried out as log probabilities
        # Should be scalar representing log marginal prob
        log_marginal_prob = logsumexp(self.alphas[:,-1] + self.emission[:, vocab_id])
        
        return log_marginal_prob


    def backward(self):
        self.betas = np.full((self.num_states, len(self.doc)), fill_value=-inf)

        # For each character, reverse order
        for char_id in np.flip(np.arange(len(self.doc))):
            # Vocab id is this character's position in the emission matrix
            vocab_id = self.vocab.get(self.doc[char_id])

            # Final time step
            if char_id == len(self.doc)-1:
                # Backward prob of last emitting hidden states is just their
                # prob of emitting the last char
                self.betas[:,char_id] = self.emission[:, vocab_id]
                continue
            
            # Adding log probabilities of the next betas, the transitions and the emission
            # Should be a self.num_states by self.num_states matrix
            # array[0,1] = log(beta1) + log(p('e'|s0)) + log(p(s1|s0))
            # From s0 to s1 ^^^
            # Need to expand dims to make broadcasting work
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            intmed = self.betas[:,char_id+1] + self.transition + e

            # Logsumexp across the rows, same as add all probs going out of a state
            self.betas[:,char_id] = logsumexp(intmed, axis=1)

        # The beta of the intial state should be the same as the log prob
        log_marginal_prob = self.betas[self.initial_state,0]
        
        return log_marginal_prob


    def update_params(self, log_marginal_prob):
        # Probability of  counts at each timestep
        uncollected = np.zeros((self.num_states, self.num_states, len(self.doc)))
        
        for char_id in np.arange(len(self.doc)):
            vocab_id = self.vocab.get(self.doc[char_id])

            # Expand alphas to be column vectors and betas to be row
            # Makes sense because alphas correspond to row index, which corresponds the orgin state
            # Betas correspond to col index, which corresponds to the destination state
            left_alphas = np.expand_dims(self.alphas[:,char_id], axis=1)
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            right_betas = np.expand_dims(self.betas[:,char_id], axis=0)

            # Add log probs of left alpha, right beta, transition, and emission prob for each transition
            # Output should be a num_states x num_states matrix for each character in text
            uncollected[:,:,char_id] = left_alphas + self.transition + e + right_betas
        
        # Normalize with marginal prob
        uncollected -= log_marginal_prob

        # Collect counts for each transition, output should be same size as self.transition
        transition_counts = logsumexp(uncollected, axis=2)

        # Update transition matrix by normalizing by row sum, since rows of transition matrices
        # Must add up to 1
        self.transition = softmax(transition_counts, axis=1)
        
        # Records indices of all instances of each letter in the text
        indices = [np.where(self.doc == letter) for letter in self.vocab.keys()]

        for vocab_id in np.arange(self.vocab.values()):
            emission_counts[:,:,vocab_id] = np.sum(uncollected, where=indices[vocab_id], axis=(0,1))

        # Result of sum should be 2x27, #TODO might need to squeeze
        self.emission = np.sum(emission_counts, axis=1)

        return 'success'

    def train(self, iters: int):

        log_probs_k = []
        emission_probs_A = []
        emission_probs_N = []

        for k in iters:
            log_prob = self.forward()
            # Plotting average log prob as function of # iters
            log_probs_k.append(log_prob / len(self.doc))
            self.backward()
            self.update_params()
            emission_probs_A.append(self.emission[:,0])
            emission_probs_N.append(self.emission[:,13])

        emission_probs_final_s0 = self.emission[0,:]
        emission_probs_final_s1 = self.emission[1,:]

        plot(emission_probs_final_s0)
        plot(emission_probs_final_s1)

        plot(log_probs_k)
        plot(emission_probs_A)
        plot(emission_probs_N)


    def test(self, iters: int):
        pass

    def save(self, destination: Path) -> None:
        import pickle
        logging.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved model to {destination}")

        return NotImplementedError


    @classmethod
    def load(cls, source: Path) -> HiddenMarkovModel:
        import pickle  # for loading/saving Python objects
        logging.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            result = pickle.load(f)
            logging.info(f"Loaded model from {source}")
            #return result

        return NotImplementedError
