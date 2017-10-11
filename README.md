# POS
pivot selection for POS tagging

## Pre-process
- *classify_pos.py*: prepare the classification data

## Pivot Selection
Unlabelled pivot selection (FREQ_U, MI_U, PMI_U, PPMI_U) between two domains (S_U, T_U)

Labelled pivot selection (FREQ_L, MI_L, PMI_L, PPMI_L) between POS category+ and POS category- ({S_L}^+, {S_L}^-)

- *pos_data.py*: unlabelled pivot selection, general labelled pivot selection (without weighting)
- *dist_pos_data.py*: labelled pivot selection using distribution for each POS category
- *f1_pos_data.py*: labelled pivot selection using F1 score (F1 score pre-computed from training data)

## Evaluation
- *pivot_wordnet.py*: how many words are nouns?
- *test_eval.py* and *roc_curve.py*: p,r,f1?
