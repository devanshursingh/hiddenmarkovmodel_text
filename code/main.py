#!/usr/bin/env python3

import argparse
from hmm import HiddenMarkovModel as hmm
import matplotlib.pyplot as plt
import random
import string

"""
This file is used to train hmm.py using the command line.
It defines the following arguments:

--train -> This is a string that forms the file path of the training data

--test -> This is a string that forms the file path of the test data

--nstates -> This integer is the number of hidden states the HMM should have

--iters -> This integer is the number of iterations the EM algorithm should run for

--altint -> This boolean flag indicates whether you want to use the alternative intialization

NOTE: Uncomment the commented lines if using 4 states, code does not account for this automatically
"""

ap = argparse.ArgumentParser()
ap.add_argument('--train', default="../data/train.txt")
ap.add_argument('--test', default="../data/test.txt")
ap.add_argument('--nstates', default=2)
ap.add_argument('--iters', default=600)
ap.add_argument('--altinit', action='store_true')

args = ap.parse_args()

# Initialize an HMM
model = hmm(num_states=args.nstates, alt_init=args.altinit, train_doc=args.train)

# Train the HMM
t = model.train(iters=args.iters, train_doc=args.train, test_doc=args.test)
# For 4 states:
#log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_a_s2, emission_probs_a_s3, emission_probs_n_s0, emission_probs_n_s1, emission_probs_n_s2, emission_probs_n_s3, emission_probs_final_s0, emission_probs_final_s1, emission_probs_final_s2, emission_probs_final_s3 = t
# For 2 states:
log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_n_s0, emission_probs_n_s1, emission_probs_final_s0, emission_probs_final_s1 = t

# Plot and save all of these results
vocab = list(string.ascii_lowercase).append('#')

# Plot of the final iteration's emission probabilities for each state
# Uncomment the commented lines if using 4 states
plt.plot(emission_probs_final_s0, color='red', label='State 0 Emissions')
plt.plot(emission_probs_final_s1, color='blue', label='State 1 Emissions')
#plt.plot(emission_probs_final_s2, color='green', label='State 2 Emissions')
#plt.plot(emission_probs_final_s3, color='purple', label='State 3 Emissions')
plt.xlabel('Letter ID')
plt.xticks(vocab, size='small')
plt.ylabel('Probabilities')
plt.title('Final Emission Probabilities For Alphabet')
plt.legend()
plt.show()
plt.savefig('../figs/final_emission_probs.png')

# Plot of the average log-likelihood as a function of # iterations
plt.plot(log_probs_k_train, color='red', label='Train')
plt.plot(log_probs_k_test, color='blue', label='Test')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title('Average Log-Likelihood of Data Over 600 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/avg_log_prob_k.png')

# Plots of the emission probabilities of letters 'a' and 'n' as a function of # iterations
plt.plot(emission_probs_a_s0, color='red', label='Given State 0')
plt.plot(emission_probs_a_s1, color='blue', label='Given State 1')
#plt.plot(emission_probs_a_s2, color='green', label='Given State 2')
#plt.plot(emission_probs_a_s3, color='purple', label='Given State 3')
plt.xlabel('Iterations')
plt.ylabel('Probabilities')
plt.title('Emission Probabilities of "a" Over 600 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/a_emission_probs.png')

plt.plot(emission_probs_n_s0, color='red', label='Given State 0')
plt.plot(emission_probs_n_s1, color='blue', label='Given State 1')
#plt.plot(emission_probs_n_s2, color='green', label='Given State 2')
#plt.plot(emission_probs_n_s3, color='purple', label='Given State 3')
plt.xlabel('Iterations')
plt.ylabel('Probabilities')
plt.title('Emission Probabilities of "n" Over 600 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/n_emission_probs.png')