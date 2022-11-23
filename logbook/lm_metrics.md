# Testing metrics based on language modeling

## Theory
First, let's describe the general framework. Consider summary $S$ of text $O$, some other text $T$ and pre-trained language model $L$.
We want to measure how informative $S$ is w.r.t. $T$. To do so we feed $S$ into $L$ without calculating any losses,
then feed $T$ there too, but now for each token we check how likely it was based on the previous ones. The idea is that 
the more informative summary we have, the smaller will be the mean loss. Let's denote the loss we get by $f(S, T, L)$.

Now, what do we take for the $T$? 

First, we can take the text itself:
$$ F_1(S, O, L) = f(S, O, L)$$
The optimal summary in such metric will contain the most important information that is needed to 
reproduce this text. And while it's already a nice property, in most cases we are not interested in memorizing the text.
We would prefer instead to get the information that will be useful later.

Consider, for example, the task of summarizing a codebase of some library. This memorizing approach will try to put into the summary, 
though in concise form, all the information about this code.
And what we would prefer here is to get the information about interfaces exposed by the library (because they will be almost certainly needed), 
and only after them something about internal implementation (in case we would like to reimplement some parts of it).

Now it's natural to think about using some continuation of the original text as $T$. If $O$ was a prefix of $W$, we can use:
$$F_2(S, O, L) = f(S, W \setminus O, L)$$
In the example, we could take some code that actually uses this library, and try to predict it. Obviously, such metric would put less emphasis on the internal details of the library. But now we have an even bigger problem: the optimal solution here is a summary of $T$. In other words, it doesn't depend on the original text at all.

Suppose we have a function that for any text $O$ returns the distribution of it's continuations $c(O)$. What if we take $T \sim c(O)$ now, and 
take the expectation of our previous metric?
$$F_3(S, O, L) = \mathbb{E}[f(S, T, L)], T \sim c(O)$$
It nicely combines the two previous approaches, because it's again a function of the original text,
and not some arbitrary continuation, and at the same time, it still forces the optimal solution to include the information that has higher chances
to be relevant to the likely continuations.

Note that we can sample from $c(O)$ by using the same language model $L$.
So for a given text $O$ we can generate some continuations $T_1, \dots, T_k$ and then take mean $f(S, T_i, L)$.

$~$

Finally, what language model $L$ should we take? The answer is trivial -- the better $L$ we take, the better metric we get. But it's important to note how
important this choise is. For example, if we take GPT-2 as $L$, the resulting metric gives better scores to a prefix of the original text than to an actual
summary of it (the text and the summary were taken from XSum dataset). But it we replace it with GPT-2 Medium (or Large), everything is fine. 
Also, if the language model has small context size, or for other reasons tends to forget long-term information, this metric will favor summaries focusing on the last sentences of the text.

## Experiments
Here the first approach ($T = O$) was tested with different versions of GPT-2. Tested summaries were:
1. "Officers searched properties in the Waterfront Park and Colonsay",
2. "A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh.",
3. " ",
4. "A piece of news."
5. "Something boring happened."
6. "The pancreas can be triggered to regenerate itself through a type of fasting diet, say US researchers."
7. "One, two, three!"
8. "#$%^&*(*&^%$#$%)"

And the text was:

"Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday."

Results:

| gpt2 | gpt2-medium | gpt2-large |
|------|-------------|------------|
| 2.69 | 2.30        | 2.30       |
| 2.78 | 2.27        | 2.26      |
| 3.32 | 2.89        | 2.93       |
| 3.33 | 2.85        | 2.91       |
| 3.41 | 2.93        | 2.99       |
| 3.41 | 3.04        | 2.98       |
| 3.53 | 3.00        | 3.09       |