import logging
import math
from pathlib import Path
from typing import Callable

import argparse

from corpus import TaggedCorpus, desupervise
from eval import eval_tagging, model_cross_entropy, model_error_rate
from hmm import HiddenMarkovModel as hhmm
from crf import HiddenMarkovModel as chmm
from lexicon import build_lexicon
import torch


ap = argparse.ArgumentParser()
ap.add_argument("eval_file")
ap.add_argument('--model', default="hmm.py")
ap.add_argument('--train', nargs='*', default=["../data/icraw"],
                help='total number of examples to train on')
ap.add_argument('--crf', action='store_true')

args = ap.parse_args()

# Set up logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Get the corpora

if len(args.train) > 1:
    entrain = TaggedCorpus(Path(args.train[0]), Path(args.train[1]))
else:
    entrain = TaggedCorpus(Path(args.train[0]), Path(args.train[0]))
ensup =   TaggedCorpus(Path(args.train[0]), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
endev =   TaggedCorpus(Path(args.eval_file), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
logging.info(f"Tagset: f{list(entrain.tagset)}")
known_vocab = TaggedCorpus(Path(args.train[0])).vocab    # words seen with supervised tags; used in evaluation

# Initialize an HMM
lexicon = build_lexicon(entrain, embeddings_file=Path('../data/words-50.txt'))  # works better with more attributes!
if args.crf:
    hmm = chmm(entrain.tagset, entrain.vocab, lexicon)
else:
    hmm = hhmm(entrain.tagset, entrain.vocab, lexicon)

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training, since we don't really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
hmm.train(corpus=ensup,   loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1)

# Now let's throw in the unsupervised training data as well, and continue
# training to try to improve accuracy on held-out development data.
# We'll stop when accuracy is getting worse, so we can get away without regularization,
# but it would be better to search for the best `reg` and other hyperparameters in this call.
loss_dev = lambda model: model_error_rate(model, eval_corpus=endev, known_vocab=known_vocab)
hmm.train(corpus=entrain, loss=loss_dev, minibatch_size=30, evalbatch_size=10000, lr=0.0001,  reg=0)

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
for m, sentence in enumerate(endev):
    if m >= 10: break
    viterbi = hmm.viterbi_tagging(desupervise(sentence), endev)
    counts = eval_tagging(predicted=viterbi, gold=sentence,
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']

    logging.info(f"Gold:    {sentence}")
    logging.info(f"Viterbi: {viterbi}")
    logging.info(f"Loss:    {denom - num}/{denom}")
    logging.info(f"Prob:    {math.exp(hmm.log_prob(sentence, endev))}")
