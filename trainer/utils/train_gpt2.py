import pandas as pd
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from torch.nn.utils import clip_grad_norm_


"""
Process the data by reading it from a CSV file, filling any missing values, and creating a new DataFrame.

Returns:
    pandas.DataFrame: Processed data with columns 'text' and 'label'.
"""


def process_data(input_dir):
    """
    load the data from source
    """

    train_data = pd.read_csv(str(input_dir) + "/appceleratorstudio.csv")
    # some rows have no description, fill blank to avoid Null
    train_data = train_data.fillna(" ")
    d = {
        "text": (train_data["title"] + ": " + train_data["description"]).tolist(),
        "label": train_data["storypoint"],
    }
    train_data = pd.DataFrame(data=d)

    # split data
    train_text = train_data["text"]
    train_labels = train_data["label"]

    return train_text, train_labels


def tokenize_text_list(text_list, tokenizer: GPT2Tokenizer, SEQUENCE_LENGTH):
    return tokenizer.batch_encode_plus(
        text_list, truncation=True, max_length=SEQUENCE_LENGTH, padding="max_length"
    )


def prepare_dataloader(train_text, train_labels, BATCH_SIZE_RATIO):
    train_text_tensor = torch.tensor(train_text["input_ids"])
    train_labels_tensor = torch.tensor(train_labels.tolist())

    tensor_dataset = TensorDataset(train_text_tensor, train_labels_tensor)
    sampler = RandomSampler(tensor_dataset)
    BATCH_SIZE = int(len(train_text["input_ids"]) * BATCH_SIZE_RATIO)
    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader


def train(
    train_dl,
    model: GPT2ForSequenceClassification,
    LEARNING_RATE,
    device,
    epochs,
    output_dir,
):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        total_train_loss = 0
        for step, batch in enumerate(train_dl):
            b_input_ids = batch[0].to(device)
            b_label_ids = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, labels=b_label_ids, return_dict=True)

            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()
            print(" *** step:", step, "loss:", loss.item())
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            del step, batch, b_input_ids, b_label_ids, loss, logits

        avg_train_loss = total_train_loss / len(train_dl)
        print(">>>>>> Average train loss:", avg_train_loss)

        torch.save(model.state_dict(), output_dir + "/gpt2sp_" + str(epoch) + ".pth")
        del avg_train_loss, total_train_loss
    return model
