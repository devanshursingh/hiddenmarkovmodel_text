import logging
import math
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import model_cross_entropy
from eval import eval_tagging
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Make an HMM with randomly initialized parameters.
icsup = TaggedCorpus(Path("../data/icsup"), add_oov=False)
logging.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
logging.info(f"Ice cream tagset: {list(icsup.tagset)}")
lexicon = build_lexicon(icsup, one_hot=True)   # one-hot lexicon: separate parameters for each word
hmm = HiddenMarkovModel(icsup.tagset, icsup.vocab, lexicon)

#hmm.printAB()
hmm.updateAB()
#hmm.printAB()

#cross_entropy_loss = lambda model: model_cross_entropy(model, icsup)
#hmm.train(corpus=icsup, loss=cross_entropy_loss,
          #minibatch_size=10, evalbatch_size=500, lr=0.01, tolerance=0.0001)

#hmm.printAB()
print("##################")

#eval.eval_tagging()

icdev = TaggedCorpus(Path("../data/icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
#assert(len(icraw)==1)   # just the single spreadsheet sentence
#for sentence in icraw:
    #logging.info(hmm.viterbi_tagging(sentence, icraw))
for sentence in icdev:
    Z = hmm.log_forward(sentence, icdev)
    #e = eval_tagging(sent, sentence, icsup.vocab)
    #print(e)
