import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import config
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class SentimentDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
        review,
        max_length=self.max_len,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors,
        truncation=True
    )

    # print (len(encoding['input_ids'].flatten()))
    return {
        'review_text': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    # print (ds[0])
    # raise Exception()
    return DataLoader(
        ds,
        batch_size=batch_size
    )

def read_data():
    df = pd.read_csv('./IMDb_Reviews.csv').head(100)
    df_train, df_test = train_test_split( df,test_size=0.1,random_state=config.RANDOM_SEED)
    df_train, df_val = train_test_split(df_train,test_size=0.1,random_state=config.RANDOM_SEED)
    return df_train, df_val, df_test

def get_data_loader():
    df_train, df_val, df_test = read_data()
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    train_dl = create_data_loader(df_train, tokenizer, config.MAX_LEN, config.TRAIN_BATCH_SIZE)
    val_dl = create_data_loader(df_val, tokenizer, config.MAX_LEN, config.VAL_BATCH_SIZE)
    test_dl = create_data_loader(df_test, tokenizer, config.MAX_LEN, config.TEST_BATCH_SIZE)
    return train_dl, val_dl, test_dl

