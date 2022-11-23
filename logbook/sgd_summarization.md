# Summarization by SGD

## Algorithm
The idea is to take a differentiable metric, like in [Testing metrics based on language modeling](lm_metrics.md), and then just optimize it w.r.t. word embeddings of the summary.

The problem is that after this process we will have to discretize this embeddings back to embeddings that exist in vocabulary, so that we can turn them
back into text. And at this step we can lose a lot of information, if the found minimum wasn't wide enough. 

Suppose our current metric is $f(x)$, then we can blur it as follows:
$$g(x) = \mathbb{E}[f(x + \varepsilon)],\quad \varepsilon \in \mathcal{N}(0, \sigma^2)$$
And to make it computable, we can take a finite number of samples, from which then take a sample mean.

## Results
LM used here was GPT-2 Medium, and optimizer was Adam. [Here](https://colab.research.google.com/drive/1wE-0SpWPKJSS5iuh3DG3R_-nNjcGm8WC?usp=sharing) is the notebook.

Text used for summarization:
"Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday."

Summaries had fixed length, and a possible summary of such length (for comparisons) was:
"Officers searched properties in the Waterfront Park and Colonsay" with loss=2.3063.

First, if we just run it without any regularization:

loss = 0.007, `medical, by of to was externalToEVA anrawdownload`

Now, let's add l2 regularization (coefficent was 0.1):

loss = 0.008, `in in, in in,, an an in in to`

So it seems that by default something like [P-tuning](https://arxiv.org/pdf/2103.10385.pdf) happens and the model gets very precise and concise summaries,
but can't put them into words (the sentences given here show the closest text
representation of the final embeddings, if we evalute loss for them, it will be nearly the same as for random string).

If we train with `noise = 0.1, l2 = 0.001`:

`police The marijuana It on had in couldActionCode�`

loss (with noise) = 0.01, loss (without noise) = 0.009, loss (after discretization) = 3.01

Here the loss after discretization is still at the random string level, but at least we can see some meaningful words in the output.

With `noise = 0.2, l2 = 0.001`:

`un in ocean police had YourawdownloadT Water Edinburgh courthouse`

loss (with noise) = 0.93, loss (after discretization) = 2.63

Finally, we have loss somewhere in between reasonable summaries (~2.3) and unrelated summaries (~2.9). Also, there are even more words from the text,
like "Edinburgh", "court", "Water"(...front) and "police" which relates to officers.

Maybe noise is all you need? `noise = 0.5, l2 = 0.001`:

`, anin an,, firearms Edinburgh searched Dram`

loss (with noise) = 1.89, loss (after discretization) = 2.66

