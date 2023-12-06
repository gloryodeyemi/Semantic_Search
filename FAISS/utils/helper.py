import torch
from transformers import AutoModel, AutoTokenizer


def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # extract the embeddings
    text_embeddings = last_hidden_states.mean(dim=1)  # mean pooling across the sequence dimension
    return text_embeddings


def load_model_and_tokenizer(model_name, tokenizer_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def load_and_return(model_name):
    # load the model and tokenizer
    if model_name == "bert-base-uncased":
        model, tokenizer = load_model_and_tokenizer(model_name=model_name,
                                                    tokenizer_name=model_name)
    elif model_name == "sbert":
        model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        model, tokenizer = load_model_and_tokenizer(model_name=model_ckpt, tokenizer_name=model_ckpt)
    elif model_name == "gpt2":
        model, tokenizer = load_model_and_tokenizer(model_name=model_name,
                                                    tokenizer_name=model_name)
        tokenizer.add_special_tokens({'pad_token': '[CLS]'})
        tokenizer.pad_token = '[CLS]'
    else:
        return
    return model, tokenizer
