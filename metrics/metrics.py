
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import numpy as np
from sentence_transformers import CrossEncoder
import sacrebleu as sacre_bleu
from sacrebleu.metrics import BLEU


def bleu(references,hypothesis):
    score = corpus_bleu([references],[hypothesis],smoothing_function = SmoothingFunction().method1)
    return score

def sacrebleu(references,hypothesis):
    bleu_score = sacre_bleu.corpus_bleu(hypothesis, references,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize="intl",
                                     use_effective_order=False)
    return bleu_score
