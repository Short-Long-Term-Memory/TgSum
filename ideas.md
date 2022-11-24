# Approaches to summarization

## Metrics
- ROUGE-like: computes some statistics based on word overlaps
- Based on word embeddings: some algebra with embeddings of individual words
- Based on usefulness: summary must be a good alternative to the original text in terms of some downstream task
    - QA-based, as [here](https://arxiv.org/pdf/1909.01610.pdf)
    - LM-based, as in [Testing metrics based on language modeling](logbook/lm_metrics.md)
        - Add attention-like model that will predict which parts of the text are most important, trained supervised or derived from the hidden states or attention layers of language models
            - Removing or permuting small parts of the text and checking how the metric changes (similar to permutation importance)
        - Reverse LM
        - Promt engineering and other tricks:
            - Some delimiter between $S$ and $T$
            - Subtracting logits from short-context prediction from the usual ones, to reinforce using long-term dependecies in text (as [here](https://arxiv.org/abs/2110.08294)) 

## Summarization algorithms
- Take a good and/or differentiable metric (as in [LM experiments](logbook/lm_metrics.md)) and optimize it, unsupervised or no training at all
    - Gradient descent, as in [Summarization by SGD](logbook/sgd_summarization.md)
    - Sample candidates from another summarization model
    - Metaheuristics
    - Diffusion on the word embedding level (as [here](https://arxiv.org/pdf/2211.04236.pdf))
    - Additional Seq2seq (for differentiable metrics) or RL (for non-differentiable) model trained on it
- Train auto-encoder, then add some latent space manipulations (as in [MeanSum](http://proceedings.mlr.press/v97/chu19b/chu19b.pdf)), unsupervised training
- Seq2seq architectures trained on texts and their summaries, supervised training
- Fine-tune existing seq2seq models to be able to interact with the user