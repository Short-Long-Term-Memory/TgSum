from torch import cuda
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from datasets import load_dataset
import os

device = 'cuda' if cuda.is_available() else 'cpu'
config = {"TRAIN_BATCH_SIZE": 2, "VALID_BATCH_SIZE": 2, "TRAIN_EPOCHS": 1, "VAL_EPOCHS": 1, "LEARNING_RATE": 1e-4,
          "SEED": 42, "MAX_LEN": 512, "SUMMARY_LEN": 150}


class SummaryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ctext = "summarize: " + self.data[index]['dialogue']
        ctext = ctext.replace('\n', '')

        text = self.data[index]['summary']
        text = text.replace('\n', '')

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.summ_len, pad_to_max_length=True,
                                                  truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(tokenizer, model, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 100 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def main(model_checkpoint):
    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    train_ds, test_ds = load_dataset('knkarthick/dialogsum', split=['train', 'test'])

    # print("TRAIN Dataset: {}".format(train_ds.shape))
    # print("TEST Dataset: {}".format(test_ds.shape))

    training_set = SummaryDataset(train_ds, tokenizer, config["MAX_LEN"], config["SUMMARY_LEN"])
    val_set = SummaryDataset(test_ds, tokenizer, config["MAX_LEN"], config["SUMMARY_LEN"])

    train_params = {
        'batch_size': config["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["LEARNING_RATE"])

    for epoch in range(config["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    weights_path = f"{model_checkpoint}_ft"
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model.save_pretrained(weights_path)

    for epoch in range(config["VAL_EPOCHS"]):
        predictions, actuals = validate(tokenizer, model, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv(f'{model_checkpoint}_{epoch}_summary_predictions.csv')


if __name__ == "__main__":
    main("t5-base")
