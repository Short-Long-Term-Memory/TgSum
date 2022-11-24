import unittest

import torch
from metric_experiments.lm import LM
from metric_experiments.metrics import optimize_summary


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

    def test_generate(self):
        lm = LM.from_pretrained("distilgpt2")
        message = "A man has appeared in court"
        
        torch.manual_seed(0)
        result1 = lm.generate_from_text(message, length=50, k=1, alpha=0.0)
        torch.manual_seed(0)
        result2 = lm.generate_from_text(message, length=50, k=1, alpha=0.9)
        torch.manual_seed(0)
        result3 = lm.generate_from_text(message, length=50, k=10, alpha=0.0)
        torch.manual_seed(0)
        result4 = lm.generate_from_text(message, length=50, k=10, alpha=0.9)

        self.assertTrue(isinstance(result1, str))
        self.assertTrue(isinstance(result2, str))
        self.assertTrue(isinstance(result3, str))
        self.assertTrue(isinstance(result4, str))
        self.assertEqual(result1, result2)
        self.assertNotEqual(result2, result3)
        self.assertNotEqual(result3, result4)