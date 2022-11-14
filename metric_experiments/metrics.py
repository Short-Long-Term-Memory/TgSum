import torch
from lm import LM


def optimize_summary(
    lm: LM,
    sum_embs: torch.Tensor,
    validation: str,
    epochs: int = 100,
    lr: float = 0.5,
    noise: float = 0.1,
    samples: int = 5,
    l2: float = 0.01,
):
    sum_embs = sum_embs.detach().requires_grad_(True)
    optim = torch.optim.Adam([sum_embs], lr=lr)
    discrete_emb = lm.discretize(sum_embs.data)

    for it in range(epochs):
        loss = l2 * (sum_embs**2).sum()
        for _ in range(samples):
            noised_embs = sum_embs + torch.randn_like(sum_embs) * noise
            loss += lm.loss_emb(noised_embs, validation) / samples
        print(f"loss = {loss.item()}, std(emb) = {sum_embs.std()}", flush=True)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cur_discrete = lm.discretize(sum_embs.data)
        if not torch.allclose(discrete_emb, cur_discrete):
            with torch.no_grad():
                discrete_emb = cur_discrete.clone()
                # summary_emb.data = cur_discrete.clone().data
                text = lm.ids_to_text(lm.embs_to_ids(sum_embs.squeeze(0)))
                print(it, text, flush=True)
        if it % 100 == 0:
            text = lm.ids_to_text(lm.embs_to_ids(sum_embs.squeeze(0)))
            print(it, text, flush=True)
    return sum_embs
