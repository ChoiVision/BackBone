import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

class Network(pl.LightningModule):
    def __init__(self, config, model):
        self.config= config
        self.model= model
        self.accuracy= torchmetrics.Accuracy()
        self.criterion= nn.CrossEntropy()
        self.value= list()

    def forward(self, x):
        x= self.model(x)
    
        return x
    
    def configure_optimizers(self):
        optimizer= optim.SGD(self.parameters(), lr=self.config.LR)
        
        return optimizer

    def training_step(self, batch, batch_index):
        return self._step(batch, step='train')

    def validation_step(self, batch, batch_index):
        return self._step(batch, step='valid')

    def test_step(self, batch, batch_index):
        image= batch
        output= self(image)
        value= torch.softmax(output, 1).detach().cpu().numpy()
        
    def _step(self, batch, step):
        image, label= batch
        output= self(image)
        loss= self.criterion(output, label)
        acc= self.accuracy(output, label)
        self.log_dict(
            {
                f'{step}_loss': loss,
                f'{step}_accuracy': acc
                },
                prog_bar= True
        )

        return loss