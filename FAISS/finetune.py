import os
import time
from utils import helper, preprocess
import torch
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset


def tokenize_data(model_n, data_, model_tokenizer, max_length=512):
    input_ids_list = []
    attention_mask_list = []

    for _, item in data_.iterrows():
        try:
            text = f"{item.title} {item.abstract}"
            # tokenize the text
            encoded_text = model_tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids_list.append(encoded_text['input_ids'])
            attention_mask_list.append(encoded_text['attention_mask'])
        except TypeError:
            print(f"TypeError: Item causing the issue: {item}")

    input_ids_tensor = torch.cat(input_ids_list, dim=0)
    attention_mask_tensor = torch.cat(attention_mask_list, dim=0)

    return input_ids_tensor, attention_mask_tensor


def finetune_and_save_model(model_name, model, model_tokenizer, preprocessed_data):
    print(f"Fine-tuning {model_name} model...")
    init_time = time.time()
    # define hyperparameters
    num_epochs = 5
    batch_size = 16
    learning_rate = 2e-5

    # define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # convert processed data to model format
        input_ids, attention_mask = tokenize_data(model_name, preprocessed_data, model_tokenizer)
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for batch in dataloader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            optimizer.zero_grad()
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state  # get the last hidden states

            # backpropagation and optimizer step
            optimizer.step()

        scheduler.step()

    # save the fine-tuned model and tokenizer
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_model_path = os.path.join(directory, f"{model_name}_finetuned.pth")
    output_tokenizer_path = os.path.join(directory, f"{model_name}_tokenizer")

    model.save_pretrained(output_model_path)
    model_tokenizer.save_pretrained(output_tokenizer_path)

    # get and save the training time
    training_time = round(time.time() - init_time, 3)
    print(f"Fine-tuning done and model saved.\nTime taken: {training_time}s")
    preprocess.model_train_time(model_name, training_time=training_time, time_to_update="training")


def main(model_type, data):
    lang_model, tokenizer = helper.load_and_return(model_type)
    # finetune the BERT model
    finetune_and_save_model(model_name=model_type, model=lang_model, model_tokenizer=tokenizer, preprocessed_data=data)


# load the dataset
train_data = preprocess.convert_to_dataframe()

# fine-tune bert model
main("bert-base-uncased", train_data)
# fine-tune gpt2 model
main("gpt2", train_data)
