import torch
import config
import torch.nn as nn
from transformers import BertModel
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(output.pooler_output)
    output = self.fc(output)
    return output


class SentimentModel(pl.LightningModule):
    def __init__(self, train_dl=None, val_dl=None, test_dl=None):
        super(SentimentModel, self).__init__()        
        self.model = SentimentClassifier(n_classes=config.N_CLASS)
        self.criterion = nn.CrossEntropyLoss()   
        self.learning_rate = 0.0001     
        self.train_dl, self.val_dl, self.test_dl = train_dl, val_dl, test_dl

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_nb):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(preds, targets)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return {'val_loss': self.criterion(preds, targets)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl