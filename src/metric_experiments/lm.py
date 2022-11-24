import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    ReformerModelWithLMHead,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput


class ReformerTokenizer(PreTrainedTokenizer):
    def __call__(self, text: str, return_tensors: str) -> torch.Tensor:
        assert return_tensors == "pt"
        if not isinstance(text, bytes):
            text = str.encode(text)
        input_ids = torch.tensor([x + 2 for x in text]).unsqueeze(0)
        return {"input_ids": input_ids}

    def decode(self, ids: torch.Tensor) -> str:
        assert ids.dim() == 1
        return "".join([chr(x - 2) if x > 1 else "" for x in ids])


class DummyModel(PreTrainedModel):
    def __init__(self, alphabet: int):
        self.vocab = alphabet + 2  # to be compatible with ReformerTokenizer
        self.device_ = "cpu"

    def to(self, device: str):
        self.device_ = device
        return self

    def __call__(
        self, inputs_embeds: torch.Tensor, labels: torch.Tensor
    ) -> CausalLMOutput:
        batch, length, _ = inputs_embeds.shape
        assert labels.shape == (batch, length)
        logits = torch.ones((batch, self.vocab, length)).to(self.device_)
        mask = (labels != -100).to(self.device_)
        losses = F.cross_entropy(logits, labels, reduction="none")
        assert losses.shape == (batch, length)
        loss = (losses * mask).sum(dim=-1) / mask.sum(dim=-1)
        return CausalLMOutput(loss, logits)


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
        print("dummy", checkpoint == "dummy")
        print("gpt", "gpt" in checkpoint)
        print("reformer", "reformer" in checkpoint)

        print("dummy", checkpoint == "dummy")
        if checkpoint == "dummy":
            alphabet, dim = 256, 10
            tokenizer = ReformerTokenizer()
            model = DummyModel(alphabet)
            embeddings = torch.randn((alphabet, dim))
            return LM(tokenizer=tokenizer, model=model, embeddings=embeddings)
        if "gpt" in checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(checkpoint)
            embeddings = model.transformer.wte.weight
            return LM(tokenizer=tokenizer, model=model, embeddings=embeddings)
        if "reformer" in checkpoint:
            tokenizer = ReformerTokenizer()
            model = ReformerModelWithLMHead.from_pretrained(checkpoint)
            embeddings = model.reformer.embeddings.word_embeddings.weight
            return LM(tokenizer=tokenizer, model=model, embeddings=embeddings)
        raise RuntimeError(f"unknown checkpoint {checkpoint}")

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
        assert ignore_ids.dim() == 1 and val_ids.dim() == 1
        labels = torch.cat((ignore_ids, val_ids)).unsqueeze(0)
        assert sum_emb.dim() == 2 and val_embs.dim() == 2  # length, dim
        inp_embs = torch.cat((sum_emb, val_embs), dim=0).unsqueeze(0)
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

    def generate_embs(self, input_ids, length, p):
        return self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=input_ids.size(1) + length,
            typical_p=p,
            do_sample=True,
        ).squeeze(0)

    def generate_from_text(self, input_text, **kwargs):
        embs = self.text_to_ids(input_text).unsqueeze(0).to(self.device)
        generated = self.generate_embs(embs, **kwargs)
        return self.ids_to_text(generated)
