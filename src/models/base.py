import lightning.pytorch as pl
import torch
import torchmetrics as tm


class BaseClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_outputs = {"pred": [], "target": []}
        self.val_outputs = {"pred": [], "target": []}
        self.test_outputs = {"pred": [], "target": []}

    def _on_epoch_end(self, outputs, stage):
        preds = torch.cat(outputs["pred"])
        targets = torch.cat(outputs["target"])
        for key in outputs.keys():
            outputs[key].clear()
        self._log_metrics(preds, targets, stage)

    def on_train_epoch_end(self):
        self._on_epoch_end(self.train_outputs, "train")

    def on_validation_epoch_end(self):
        self._on_epoch_end(self.val_outputs, "test")

    def on_test_epoch_end(self):
        self._on_epoch_end(self.test_outputs, "best_test")

    def _log_metrics(self, preds, targets, stage):
        metrics_dict = {}
        for name, fn in self.metrics.items():
            value = fn(preds, targets)
            self.log(f"{stage}/{name}", value, prog_bar=True)

    def _step(self, batch):
        if self.use_attention_mask:
            x, attention_mask, y = batch
            y_hat = self(x, attention_mask=attention_mask)
        else:
            x, y = batch
            y_hat = self(x)
        return y, y_hat

    def training_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss.detach(), prog_bar=True)
        self.train_outputs["pred"].append(y_hat.detach().cpu())
        self.train_outputs["target"].append(y.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        self.val_outputs["pred"].append(y_hat.detach().cpu())
        self.val_outputs["target"].append(y.detach().cpu())

    def test_step(self, batch, batch_idx):
        y, y_hat = self._step(batch)
        self.test_outputs["pred"].append(y_hat.detach().cpu())
        self.test_outputs["target"].append(y.detach().cpu())
