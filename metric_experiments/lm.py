import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput


class LM:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        embeddings: torch.Tensor,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.embeddings = embeddings.detach().clone().to(self.device)

    @staticmethod
    def from_pretrained(checkpoint: str):

        if checkpoint == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            embeddings = model.transformer.wte.weight
            return LM(tokenizer=tokenizer, model=model, embeddings=embeddings)
        raise RuntimeError(f"unknown checkpoint {str}")

    def text_to_ids(self, text: str) -> torch.Tensor:
        result = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        assert result.dim() == 1
        return result

    def ids_to_text(self, ids: torch.Tensor) -> str:
        assert ids.dim() == 1
        return self.tokenizer.decode(ids)

    def emb_to_id(self, emb: torch.Tensor, metric: str = "l2") -> int:
        assert emb.dim() == 1
        if metric == "l2":
            A = ((self.embeddings - emb.unsqueeze(0)) ** 2).sum(dim=1)
            return A.argmin().item()
        if metric == "cos":
            A = emb.unsqueeze(0) @ self.embeddings.T
            return A.argmax().item()
        raise RuntimeError(f"unknown metric {metric}")

    def embs_to_ids(self, embs: torch.Tensor, metric: str = "l2") -> torch.Tensor:
        assert embs.dim() == 2  # length, embedding
        return torch.tensor([self.emb_to_id(emb, metric) for emb in embs])

    def ids_to_embs(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[ids]

    def discretize(self, embs: torch.Tensor) -> torch.Tensor:
        return self.ids_to_embs(self.embs_to_ids(embs))

    def _run_with_validation(
        self, sum_emb: torch.Tensor, val_ids: torch.Tensor
    ) -> CausalLMOutput:
        val_ids = val_ids.to(self.device)
        ignore_ids = torch.full(sum_emb.shape[:-1], -100, device=self.device)
        val_embs = self.embeddings[val_ids]
        labels = torch.vstack((ignore_ids, val_ids)).unsqueeze(0)
        inp_embs = torch.vstack((sum_emb, val_embs)).unsqueeze(0)
        return self.model(inputs_embeds=inp_embs, labels=labels)

    def losses_emb(self, summary_embs: torch.Tensor, validation: str) -> torch.Tensor:
        val_ids = self.text_to_ids(validation)
        output = self._run_with_validation(summary_embs, val_ids)
        logits = output.logits[:, -val_ids.size(1) - 1 : -1].squeeze(0)
        return F.cross_entropy(logits, val_ids, reduction="none")

    def loss_emb(self, summary_embs: torch.Tensor, validation: str) -> torch.Tensor:
        output = self._run_with_validation(summary_embs, self.text_to_ids(validation))
        return output.loss

    def losses_str(self, summary: str, validation: str) -> torch.Tensor:
        summary_embs = self.embeddings[self.text_to_ids(summary)]
        return self.losses_emb(summary_embs, validation)

    def loss_str(self, summary: str, validation: str) -> torch.Tensor:
        summary_embs = self.embeddings[self.text_to_ids(summary)]
        return self.loss_emb(summary_embs, validation)

    def visualize_lm(self, summary: str, validation: str):
        sum_ids, val_ids = map(self.text_to_ids, (summary, validation))
        sum_embs = self.ids_to_embs(sum_ids)
        output = self._run_with_validation(sum_embs, val_ids)
        x = val_ids.squeeze(0)
        y = output.logits.squeeze(0).argmax(dim=-1)[-x.size(0) - 1 : -1]
        assert len(x) == len(y)
        for i in range(len(x)):
            prefix = self.ids_to_text(x[: i + 1])
            suffix = self.ids_to_text(y[i : i + 1])
            print(prefix, "|", suffix)
