import logging
import math
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import model_cross_entropy
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Make an HMM with randomly initialized parameters.
icsup = TaggedCorpus(Path("../data/icsup"), add_oov=False)
logging.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
logging.info(f"Ice cream tagset: {list(icsup.tagset)}")
lexicon = build_lexicon(icsup, one_hot=True)   # one-hot lexicon: separate parameters for each word
hmm = HiddenMarkovModel(icsup.tagset, icsup.vocab, lexicon)
logging.info("*** Current A, B matrices (computed by softmax from small random parameters)")
#hmm.printAB()

# While training on ice cream, we will just evaluate the cross-entropy
# on the training data itself (icsup), since we are interested in watching it improve.
logging.info("*** Supervised training on icsup")
cross_entropy_loss = lambda model: model_cross_entropy(model, icsup)
hmm.train(corpus=icsup, loss=cross_entropy_loss,
          minibatch_size=10, evalbatch_size=500, lr=0.01, tolerance=0.0001, save_path=Path("ic_hmm.pkl"))

logging.info("*** A, B matrices after training on icsup (should approximately match initial params on spreadsheet [transposed])")
hmm.printAB()

# Since we used a low tolerance, that should have gotten us about up to the
# initial parameters on the spreadsheet.  Let's tag the spreadsheet "sentence"
# (that is, the sequence of ice creams) using the Viterbi algorithm.
logging.info("*** Viterbi results on icraw")
icraw = TaggedCorpus(Path("../data/icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
assert(len(icraw)==1)   # just the single spreadsheet sentence
for sentence in icraw:
    logging.info(hmm.viterbi_tagging(sentence, icraw))

# Now let's use the forward algorithm to see what the model thinks about
# the probability of the spreadsheet "sentence."
logging.info("*** Forward algorithm on icraw (should approximately match iteration 0 on spreadsheet)")
for sentence in icraw:
    prob = math.exp(hmm.log_prob(sentence, icraw))
    logging.info(f"{prob} = p({sentence})")

# Finally, let's reestimate on the icraw data, as the spreadsheet does.
logging.info("*** Reestimating on icraw (perplexity should improve on every iteration)")
negative_log_likelihood = lambda model: model_cross_entropy(model, icraw)  # evaluate on icraw itself
hmm.train(corpus=icraw, loss=negative_log_likelihood,
          minibatch_size=10, evalbatch_size=500, lr=0.001, tolerance=0.0001, save_path=Path("ic_hmm_raw.pkl"))

logging.info("*** A, B matrices after reestimation on icraw (SGD, not EM, but still should approximately match final params on spreadsheet [transposed])")
hmm.printAB()
