import dataset
import models
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.01,
   patience=7,
   verbose=False,
   mode='min',
)

def run():
    train_dl, val_dl, test_dl = dataset.get_data_loader()
    model = models.SentimentModel(train_dl, val_dl, test_dl)
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS,
                        gpus=config.GPUS, 
                        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCHS,
                        auto_lr_find=True,
                        callbacks=[early_stop_callback, checkpoint_callback])    
    trainer.fit(model)
if __name__ == '__main__':
    run()