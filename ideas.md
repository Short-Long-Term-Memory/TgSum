# Approaches to summarization
- Train auto-encoder, then add some latent space manipulations (as in [MeanSum](http://proceedings.mlr.press/v97/chu19b/chu19b.pdf)), unsupervised
- Add attention-like model that will predict which parts of the text are most important, trained supervised or derived from the hidden states or attention layers of language models
- Seq2seq architectures trained on texts and their summaries, supervised