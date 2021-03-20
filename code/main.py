#!/usr/bin/env python3

import argparse
from hmm import HiddenMarkovModel as hmm
import matplotlib.pyplot as plt
import random

ap = argparse.ArgumentParser()
ap.add_argument('--train', default="../data/train.txt")
ap.add_argument('--test', default="../data/test.txt")
ap.add_argument('--nstates', default=2)
ap.add_argument('--iters', default=3)
ap.add_argument('--altinit', action='store_true')

args = ap.parse_args()

# Initialize an HMM
model = hmm(num_states=args.nstates, alt_init=args.altinit, train_doc=args.train)

# Train the HMM
t = model.train(iters=args.iters, train_doc=args.train, test_doc=args.test)
log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_n_s0, emission_probs_n_s1, emission_probs_final_s0, emission_probs_final_s1 = t

# Plot and save all of these results
vocab = list(string.ascii_lowercase).append('#')
n1 = random.randint(0,1000)
n2 = random.randint(0,1000)
n3 = random.randint(0,1000)
n4 = random.randint(0,1000)

plt.plot(emission_probs_final_s0, color='red', label='State 0 Final Emissions')
plt.plot(emission_probs_final_s1, color='blue', label='State 1 Final Emissions')
plt.xlabel('Letters')
plt.xticks(range(27), vocab, size='small')
plt.ylabel('Probabilities')
plt.legend()
plt.show()
plt.savefig(f'../figs/final_emission_probs_{n1}.png', n1)

plt.plot(log_probs_k_train, color='red', label='Average Log Prob for Train')
plt.plot(log_probs_k_test, color='blue', label='Average Log Prob for Test')
plt.xlabel('Iterations')
plt.xticks(range(args.iters))
plt.ylabel('Average Log Prob of Data')
plt.legend()
plt.show()
plt.savefig(f'../figs/avg_log_prob_k_{n2}.png', n2)

plt.plot(emission_probs_a_s0, color='red', label='Emission Prob of a Given State 0')
plt.plot(emission_probs_a_s1, color='blue', label='Emission Prob of a Given State 1')
plt.xlabel('Iterations')
plt.xticks(range(args.iters))
plt.ylabel('Probabilities')
plt.legend()
plt.show()
plt.savefig(f'../figs/a_emission_probs_{n3}.png', n3)

plt.plot(emission_probs_n_s0, color='red', label='Emission Prob of n Given State 0')
plt.plot(emission_probs_n_s1, color='blue', label='Emission Prob of n Given State 1')
plt.xlabel('Iterations')
plt.xticks(range(args.iters))
plt.ylabel('Probabilities')
plt.legend()
plt.show()
plt.savefig(f'../figs/n_emission_probs_{n4}.png', n4)