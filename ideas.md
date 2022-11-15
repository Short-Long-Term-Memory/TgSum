# Approaches to summarization

## Metrics
- ROUGE-like: computes some statistics based on word overlaps
- Based on word embeddings: some algebra with embeddings of individual words
- Based on usefulness: summary must be a good alternative to the original text in terms of some downstream task
    - QA-based, as [here](https://arxiv.org/pdf/1909.01610.pdf)
    - LM-based, as in [our experiments](logbook/lm_metrics.md)
        - Add attention-like model that will predict which parts of the text are most important, trained supervised or derived from the hidden states or attention layers of language models
            - Removing or permuting small parts of the text and checking how the metric changes (similar to permutation importance)

## Summarization algorithms
- Take a good and/or differentiable metric (as in [LM experiments](logbook/lm_metrics.md)) and optimize it, **no training**
    - Gradient descent
    - Metaheuristics
    - Additional RL model trained on it
- Train auto-encoder, then add some latent space manipulations (as in [MeanSum](http://proceedings.mlr.press/v97/chu19b/chu19b.pdf)), unsupervised training
- Seq2seq architectures trained on texts and their summaries, supervised training