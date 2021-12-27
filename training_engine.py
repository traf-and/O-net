import glob

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from model.model import ONet
from create_data.image_data import ONetDataset

class LightningEngine(pl.LightningModule):
    def __init__(
            self,
            model: ONet,
            criterion: torch.nn.Module,
            optimizer: optim
    ):
        super(LightningEngine, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        images, target_landmark, scale, sq1, sq0 = batch
        pred_landmark = self.model(images)
        pred_landmark = self.correct_marks(pred_landmark, sq1, sq0, scale)
        
        loss = self.criterion(pred_landmark.float(), target_landmark.float())
        
        self.log('train/loss', loss.cpu().item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, target_landmark, scale, sq1, sq0 = batch
        pred_landmark = self.model(images)
        pred_landmark = self.correct_marks(pred_landmark, sq1, sq0, scale)
        
        loss = self.criterion(pred_landmark, target_landmark)
        
        self.log('valid/loss', loss.cpu().item())
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def correct_marks(self, marks, sq1, sq0, scale):
        for i in range(len(marks)):
            marks[i,:,0] += sq1[i]/scale[1][i]
            marks[i,:,1] += sq0[i]/scale[0][i]
            marks[i,:,0] *= scale[1][i]
            marks[i,:,1] *= scale[0][i]
        return marks


if __name__ == '__main__':
    file_list = sorted([i[:-3] for i in glob.glob('/content/data/landmarks_task/*/train/*.jpg')])
    dataset = ONetDataset(file_list, 'train')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.mode = 'val'
    train_data = DataLoader(train_dataset, batch_size=10, num_workers=0, shuffle = True, pin_memory=False)
    val_data = DataLoader(val_dataset, batch_size=10, num_workers=0, shuffle = False, pin_memory=False)

    net = ONet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    model = LightningEngine(net, criterion=criterion, optimizer=optimizer)
    checkpoint_callback = ModelCheckpoint(dirpath='pl_ckpt/')
    logger = TensorBoardLogger("tb_logs/", name="my_model")
    trainer = pl.Trainer(gpus=0, callbacks=[checkpoint_callback], logger=logger)

    trainer.fit(model, train_data, val_dataloaders = val_data, max_epochs=20)