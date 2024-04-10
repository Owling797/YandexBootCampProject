from pickle import dump, load
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os
import requests
from torchtext.vocab import build_vocab_from_iterator


### Self-made NN ###
# Шаг 4: Создание модели
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class VerseGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_p):
        super(VerseGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoder(x)
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output


def load_model_checkpoint(path, vocab_length):
    checkpoint = download_model(path)
    model = VerseGenerator(vocab_size=vocab_length, embedding_dim=100, hidden_dim=256, num_layers=2, dropout_p=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def generate_verse(model, start_token, length, vocab, device, temperature):
    model.eval()
    verse = [start_token]
    for _ in range(length):
        input = torch.tensor([verse[-1]])
        output = model(input)
        # Применение температуры к выходным данным
        output = output / temperature
        probabilities = torch.nn.functional.softmax(output, dim=2)
        next_token = torch.multinomial(probabilities.view(-1), num_samples=1)
        verse.append(next_token.item())
    return ' '.join(vocab.get_itos()[token] for token in verse)


### Tuned RuGPT ###
def open_and_preprocess_data(folder_path="data/texts"):
    train_path = 'data/train_dataset.txt'
    with open(train_path, "w", encoding="utf-8") as train_file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    train_file.write(file.read() + '\n')

    with open(train_path, 'r') as f:
        lines = f.readlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    n = 480
    non_empty_lines = non_empty_lines[:n]

    train_path = 'data/train_dataset_stripped.txt'
    with open(train_path, 'w', encoding="utf-8") as f:
        f.writelines(non_empty_lines)
        
    return train_path


def get_tokenizer():
    model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    '''model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)
    
    # Создание датасета
    train_dataset = TextDataset(tokenizer=tokenizer,file_path=train_path,block_size=64)
    
    # Создание даталодера (нарезает текст на оптимальные по длине куски)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm=False)
    training_args = TrainingArguments(
        output_dir="./finetuned", # The output directory
        overwrite_output_dir=True, # Overwrite the content of the output dir
        num_train_epochs=200, # number of training epochs
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=10, # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=16, # to make "virtual" batch size larger
        fp16=True,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers = (torch.optim.AdamW(model.parameters(),lr=1e-5), None)
    )
    
    trainer.train()
    
    file_path = "model_rugpt3large.pkl"
    with open(file_path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {file_path}")
    '''
    return tokenizer


def download_model(model_name):
    download_url = f"https://storage.yandexcloud.net/stud.camp.models/{model_name}"
    local_path = f'models/{model_name}'
    response = requests.get(download_url)
    if response.status_code != 200:
        print('Failed to download the file.')
        return ""
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print('File downloaded successfully.')
    with open(local_path, "rb") as file:
        model = load(file)
    return model


def generate_by_gpt(model_name, prompt):
    tokenizer = get_tokenizer()
    model = download_model(model_name)
        
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids,
                            do_sample=True,
                            num_beams=2,
                            temperature=3.5,
                            top_p=0.9,
                            max_length=128,
                            )

    return list(map(tokenizer.decode, out))[0]


def generate_by_rnn():
    tokens = download_model('tokens.pkl')
    vocab = build_vocab_from_iterator(tokens)
    model, optimizer, start_epoch, loss = load_model_checkpoint('RMG_checkpoint.pkl', len(vocab))
    start_token = vocab['<START>']  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return generate_verse(model, start_token, 30, vocab, device, temperature=0.5)


def load_model_and_generate(model_name, prompt):
    if model_name == "model_rugpt3large_gpt2_based.pkl":
        generated_text = generate_by_gpt(model_name, prompt)
        
    elif model_name == "RMG_checkpoint.pkl":
        generated_text = generate_by_rnn()
    
    else:
        generated_text = "other"

    return generated_text
