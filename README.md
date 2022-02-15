# Explainable Machine Learning in Linguistics and Applied NLP: Two Case Studies of Norwegian Dialectometry and Sexism Detection in French Tweets

## Abstract

This thesis presents an exploration of explainable machine learning in the context of a traditional linguistic area (dialect classification) and an applied task (sexism detection).

In both tasks, the input features deemed especially relevant for the classification form meaningful groups that fit in with previous research on the topic, although not all such features are easy to understand or provide plausible explanations.

In the case of dialect classification, some important features show that the model also learned patterns that are not typically presented by dialectologists.
For both case studies, I use LIME ([Ribeiro et al., 2016](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)) to rank features by their importance for the classification.

For the sexism detection task, I additionally examine attention weights, which produce feature rankings that are in many cases similar to the LIME results but that are over all worse at showcasing tokens that are especially characteristic of sexist tweets.

## Code

To re-run the experiments I carried out in my thesis, run the following scripts:
```
## LIME for dialect classification
dialects_predict.sh
# When the previous script has finished running, run:
dialects_analyze.sh

## LIME for tweet classification
tweets_predict.sh
# When the previous script has finished running, run:
tweets_analyze.sh

## Attention weights for tweet classification
tweets_predict_attention.sh
# When the previous script has finished running, run:
tweets_analyze_attention.sh
```

The exact state of the codebase as used in my MA thesis is available via the [`ma-thesis` release](https://github.com/verenablaschke/ma-thesis/releases/tag/ma-thesis).

The tables with the results are in the [models](https://github.com/verenablaschke/ma-thesis/tree/main/models) directory.

## Thesis

The full description and analysis of this research can be found in [my thesis](https://github.com/verenablaschke/ma-thesis/blob/main/thesis.pdf).
