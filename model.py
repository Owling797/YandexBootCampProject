from pickle import dump, load
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os
import requests



def split_data(df: pd.DataFrame):
    y = df['Survived']
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    return X, y

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


def load_model_and_generate(prompt, tokenizer):
    file_id = '1gIDQlNkrGToqN8nrjtWrmtGLDvioheLh'
    download_url = f'https://drive.google.com/uc?id={file_id}'
    local_path = 'data/model_rugpt3large.pkl'
    
    response = requests.get(download_url)
    if response.status_code != 200:
        print('Failed to download the file.')
        return ""
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print('File downloaded successfully.')

    with open(local_path, "rb") as file:
        model = load(file)

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

    generated_text = list(map(tokenizer.decode, out))[0]

    return generated_text
