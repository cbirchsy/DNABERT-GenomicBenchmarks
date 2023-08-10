import lightning.pytorch as pl
import torch
import torchmetrics as tm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, IA3Config, TaskType
from src.models.base import BaseClassifier


class NucleotideTransformerClassifier(BaseClassifier):
    def __init__(
        self,
        optimizer,
        scheduler,
        model_path,
        num_labels=2,
        use_attention_mask=False,
        peft=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_attention_mask = self.hparams.use_attention_mask
        self.peft = peft
        if self.peft:
            peft_config = IA3Config(
                peft_type="IA3",
                target_modules=["value", "key", "dense"],
                feedforward_modules=["dense"],
                modules_to_save=["classifier"],
            )
            self.model = get_peft_model(
                AutoModelForSequenceClassification.from_pretrained(
                    self.hparams.model_path, num_labels=self.hparams.num_labels
                ),
                peft_config,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.model_path, num_labels=self.hparams.num_labels
            )

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            "Acc": tm.Accuracy(task="multiclass", num_classes=self.hparams.num_labels),
            "F1": tm.F1Score(task="multiclass", num_classes=self.hparams.num_labels),
            "AUROC": tm.AUROC(task="multiclass", num_classes=self.hparams.num_labels),
        }

    def configure_optimizers(self):
        if self.peft:
            optimizer = self.hparams.optimizer(
                filter(lambda p: p.requires_grad, self.parameters())
            )
        else:
            optimizer = self.hparams.optimizer(self.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer),
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def forward(self, input_ids, attention_mask=None):
        if attention_mask == None:
            logits = self.model(input_ids).logits
        else:
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        return logits
