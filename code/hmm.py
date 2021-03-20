#!/usr/bin/env python3

import numpy as np
from softmax import softmax
from scipy.special import logsumexp
from math import inf, log, exp, sqrt
import math
import random
import string

class HiddenMarkovModel():

    def __init__(self, num_states: int, alt_init: bool, train_doc: str) -> None:
        # Number of hidden states
        self.num_states = num_states

        # Create dictionary that maps alphabet & ' ' to index 0-27 for emission matrix
        alphabet = list(string.ascii_lowercase) + [' ']
        vocab_index = [num for num in range(27)]
        self.vocab = {alphabet[i] : vocab_index[i] for i in range(len(vocab_index))}

        with open(train_doc, 'r') as f:
            train_doc = f.read()
            train_doc = list(train_doc)
        
        # Create and initialize params
        self.init_params(alt=alt_init, doc=train_doc)
        # Choose what initial hidden state will be with equal probability
        self.initial_state = random.choice([state for state in range(self.num_states)])


    def init_params(self, alt: bool, doc):
        if alt:
            # Alternative initialization of emission matrix
            indices = [np.where(np.array(doc) == letter)[0] for letter in self.vocab.keys()]
            b = np.full((len(doc)), False)
            rel_freq = []
            for vocab_id in np.arange(self.vocab.values()):
                rel_freq.append(np.sum(b[indices[vocab_id]]) / len(doc))
            rel_freq = np.array(rel_freq)

            randoms = np.array(random.sample(range(len(doc)), len(self.vocab)))
            randoms = randoms - np.mean(randoms)

            emission_s0 = rel_freq - l * randoms
            emission_s1 = rel_freq + l * randoms
            self.emission = np.log(np.concatenate((emission_s0, emission_s1), axis=0))
        else:
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


    def forward(self, doc):
        self.alphas = np.full((self.num_states, len(doc)), fill_value=-inf)

        # For each character
        for char_id in np.arange(len(doc)):
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
            vocab_id = self.vocab.get(doc[char_id])

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
        vocab_id = self.vocab.get(doc[char_id])

        # Final sum of all alpha probs multiplied by the emission probs of the last char
        # Carried out as log probabilities
        # Should be scalar representing log marginal prob
        log_marginal_prob = logsumexp(self.alphas[:,-1] + self.emission[:, vocab_id])
        
        return log_marginal_prob


    def backward(self, doc):
        self.betas = np.full((self.num_states, len(doc)), fill_value=-inf)

        # For each character, reverse order
        for char_id in np.flip(np.arange(len(doc))):
            # Vocab id is this character's position in the emission matrix
            vocab_id = self.vocab.get(doc[char_id])

            # Final time step
            if char_id == len(doc)-1:
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


    def update_params(self, log_marginal_prob, doc) -> None:
        # Probability of  counts at each timestep
        uncollected = np.zeros((self.num_states, self.num_states, len(doc)))
        
        for char_id in np.arange(len(doc)):
            vocab_id = self.vocab.get(doc[char_id])

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
        indices = [np.where(np.array(doc) == letter)[0] for letter in self.vocab.keys()]
        b = np.full((len(doc)), False)

        # For each letter in the alphabet, sum all transition counts that emit that letter
        for vocab_id in np.arange(self.vocab.values()):
            emission_counts[:,:,vocab_id] = np.sum(uncollected, where=b[indices[vocab_id]], axis=2)

        # Result of sum should be 2x27
        self.emission = np.sum(emission_counts, axis=1)


    def train(self, iters: int, train_doc: str, test_doc: str):
        with open(train_doc, 'r') as f:
            train_doc = f.read()
            train_doc = list(train_doc)

        with open(test_doc, 'r') as f:
            test_doc = f.read()
            test_doc = list(test_doc)

        # Record average log prob per iteration k for plotting
        log_probs_k_train = []
        log_probs_k_test = []
        # Record emission probabilities for all states for letter a for each state
        emission_probs_a_s0 = []
        emission_probs_a_s1 = []
        # For letter n
        emission_probs_n_s0 = []
        emission_probs_n_s1 = []

        # Run the EM algorithm for k iterations
        for k in range(iters):
            log_prob1 = self.forward(doc=train_doc)
            log_prob_test = self.forward(doc=test_doc)
            log_prob2 = self.backward(doc=train_doc)
            assert (log_prob1 == log_prob2) # both marginals should be equal
            self.update_params(log_marginal_prob=log_prob1, doc=train_doc)

            log_probs_k_train.append(log_prob1 / len(train_doc))
            log_probs_k_test.append(log_prob_test / len(test_doc))
            emission_probs_a_s0.append(np.exp(self.emission[0,0])[0])
            emission_probs_a_s1.append(np.exp(self.emission[1,0])[0])
            emission_probs_n_s0.append(np.exp(self.emission[0,13])[0])
            emission_probs_n_s1.append(np.exp(self.emission[1,13])[0])

        # Record the final params for each state
        emission_probs_final_s0 = np.exp(self.emission[0,:]).tolist()
        emission_probs_final_s1 = np.exp(self.emission[1,:]).tolist()

        return log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_n_s0, emission_probs_n_s1, emission_probs_final_s0, emission_probs_final_s1