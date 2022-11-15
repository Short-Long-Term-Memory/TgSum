import unittest

import torch
from lm import LM
from metrics import optimize_summary


class TestLM(unittest.TestCase):
    def test_loss_smoke(self):
        lm = LM.from_pretrained("dummy")
        summaries = [
            """Officers searched properties in the Waterfront Park and Colonsay""",
            """A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh.""",
            """ """,
            """A piece of news.""",
            """Something boring happened.""",
            """The pancreas can be triggered to regenerate itself through a type of fasting diet, say US researchers."""
            """One, two, three!""",
            """#$%^&*(*&^%$#$%)""",
        ]
        validation = """Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday."""
        for summary in summaries:
            loss = lm.loss_str(summary, validation).item()
            self.assertGreater(loss, 0)
            self.assertLess(loss, 10)

    def test_summary_optimization_smoke(self):
        lm = LM.from_pretrained("dummy")
        initial = """Officers searched properties in the Waterfront Park and Colonsay"""
        validation = """Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday."""

        sum_embs = lm.ids_to_embs(lm.text_to_ids(initial))
        self.assertEqual(lm.ids_to_text(lm.embs_to_ids(sum_embs)), initial)
        new_embs = optimize_summary(lm, sum_embs, validation, epochs=3)
        self.assertFalse(torch.allclose(sum_embs, new_embs))
