# Summarization by filtering outputs of existing models

## Algorithm
As a baseline model we used T5 fine-tuned on DialogSum dataset. During inference we would run it several times on the same input,
which gives us a few summaries to choose from. Then we calculate LM-based metric (similar to BLANC) and choose the best summary according to them.

## Results
You can see how the baseline and filtered versions work on some random samples from DialogSum [here](samples.txt). In this experiment the best of 15 summaries were chosen in each inference.
The difference might be not very noticeable just by looking, so here are some metrics:

|          | BLANC | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|-------|---------|---------|---------|
| baseline | 2.06  | 0.362   | 0.104   | 0.286   |
| filtered | 2.12  | 0.378   | 0.128   | 0.296   |

As expected, BLANC is slightly better for filtered version. But note that ROUGE scores also improved, which might mean that we are not just overfitting
for some strange metric, but actually improving the summary quality.