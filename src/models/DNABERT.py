import lightning.pytorch as pl
import torch
import torchmetrics as tm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

class BaseClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.train_outputs = {'pred':[],  'target':[]}
        self.val_outputs = {'pred':[],  'target':[]}
        self.test_outputs = {'pred':[],  'target':[]}
    
    def _on_epoch_end(self, outputs, stage):
        preds = torch.cat(outputs['pred'])
        targets = torch.cat(outputs['target'])
        for key in outputs.keys():
            outputs[key].clear()
        self._log_metrics(preds, targets, stage)

    def on_train_epoch_end(self):
        self._on_epoch_end(self.train_outputs, 'train')

    def on_validation_epoch_end(self):
        self._on_epoch_end(self.val_outputs, 'test')
        
    def on_test_epoch_end(self):
        self._on_epoch_end(self.test_outputs, 'best_test')

    def _log_metrics(self, preds, targets, stage):
        metrics_dict = {}
        for name, fn in self.metrics.items():
            value = fn(preds, targets)
            self.log(f'{stage}/{name}', value, prog_bar=True)
            
    def _step(self, batch):
        if self.use_attention_mask:
            x, attention_mask, y = batch
            y_hat = self(x, attention_mask = attention_mask)
        else: 
            x, y = batch
            y_hat = self(x)
        return y, y_hat
            
    def training_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        loss = self.loss(y_hat, y)
        self.log('train/loss', loss.detach(), prog_bar=True)
        self.train_outputs['pred'].append(y_hat.detach().cpu())
        self.train_outputs['target'].append(y.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        self.val_outputs['pred'].append(y_hat.detach().cpu())
        self.val_outputs['target'].append(y.detach().cpu())
        
    def test_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        self.test_outputs['pred'].append(y_hat.detach().cpu())
        self.test_outputs['target'].append(y.detach().cpu())

class DNABERTClassifier(BaseClassifier):
    def __init__(self, optimizer, scheduler, model_path, num_labels=2, use_attention_mask=False):
        super().__init__()
        self.save_hyperparameters()
        
        self.use_attention_mask = self.hparams.use_attention_mask

        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.model_path, num_labels=self.hparams.num_labels)

        self.loss = torch.nn.CrossEntropyLoss()
        
        self.metrics = {
            'Acc':tm.Accuracy(task='multiclass', num_classes=self.hparams.num_labels),
            'F1':tm.F1Score(task='multiclass', num_classes=self.hparams.num_labels),
            'AUROC':tm.AUROC(task='multiclass', num_classes=self.hparams.num_labels),
        }
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = {
            'scheduler':self.hparams.scheduler(optimizer),
            'name': 'learning_rate'}
        return [optimizer], [scheduler]

    def forward(self, input_ids, attention_mask=None):
        if attention_mask == None:
            logits = self.model(input_ids).logits
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits