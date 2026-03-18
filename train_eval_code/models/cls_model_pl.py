from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions, logging
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import numpy as np
import hydra
import wandb
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.cli import instantiate_class

logger = logging.get_logger(__name__)

# torch.nn.TransformerEncoderLayer
torch.optim.lr_scheduler.LinearLR

class ClsModelWithHyena(pl.LightningModule):
    def __init__(self, hyena_model, head, num_seqs, num_classes, optimizer_config, scheduler_config):
        super(ClsModelWithHyena, self).__init__()
        self.hyena_model = hydra.utils.instantiate(hyena_model)
        self.hyena_model.eval()
        self.head = hydra.utils.instantiate(head)
        if head._target_ == "torch.nn.LSTM":
            self.classifier = nn.Linear(head.hidden_size, num_classes)
        elif head._target_ == "torch.nn.TransformerEncoder":
            self.classifier = nn.Linear(hyena_model.d_model, num_classes)
        self.num_seqs = num_seqs
        self.cls_embedding = torch.nn.Parameter(torch.randn(1, 1, hyena_model.d_model))

        self.num_classess = num_classes
        self.class_names = [str(i) for i in range(num_classes)]
         
        # Initialize weights and apply final processing
        # self.post_init()

        self.metrics = {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
                        "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro"),
                        "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro"),
                        "F1 score": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")}
                        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.test_preds = []
        self.test_targets = []
       
    def forward(
        self,
        batch
    ):
        input_ids_of_seqs = batch["input_ids_of_seqs"]
        batch_size, num_seqs, num_base = input_ids_of_seqs.shape
        batch_seq_embeds = self.hyena_model(input_ids_of_seqs.view(-1, num_base))[0][:, -1, :]
        batch_seq_embeds = batch_seq_embeds.view(batch_size, num_seqs, -1)    # (bs, n_seqs, 128)
        batch_seq_embeds = torch.cat([self.cls_embedding.repeat(batch_size, 1, 1), 
                                      batch_seq_embeds], dim=1)
        if isinstance(self.head, torch.nn.TransformerEncoder):
            logits = self.head(batch_seq_embeds)[:, 0, :]      # self.head(batch_seq_embeds): (bs, 1 + n_seqs, 128)
        elif isinstance(self.head, torch.nn.LSTM):
            logits = self.head(batch_seq_embeds)[0][:, 0, :]   # self.head(batch_seq_embeds)[0]: (bs, 1 + n_seqs, 128)
        logits = self.classifier(logits)   # (bs, n_classes)
        scores = F.softmax(logits, dim=-1)

        return scores

    def configure_optimizers(self):
        for name, param in self.named_parameters():
            if "hyena_model" in name:
                param.requires_grad = False
        optimizer = instantiate_class(filter(lambda p: p.requires_grad, self.parameters()), 
                                      self.optimizer_config)
        scheduler = instantiate_class(optimizer, self.scheduler_config)
        return [optimizer], [scheduler]
    
    def on_train_batch_start(self, batch, batch_idx):
        self.log_dict({"train/lr": self.trainer.optimizers[0].param_groups[0]['lr']})

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.test_preds, axis=0)
        targets = np.concatenate(self.test_targets, axis=0)
        self.log_confusion_matrix(preds, targets)
        self.test_preds = []
        self.test_targets = []
        
 
    def on_test_epoch_end(self):
        preds = np.concatenate(self.test_preds, axis=0)
        targets = np.concatenate(self.test_targets, axis=0)
        self.log_confusion_matrix(preds, targets)
        self.test_preds = []
        self.test_targets = []

    def log_confusion_matrix(self, preds,  targets):
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classess)))
        self.logger.experiment.log({"confusion_matrix": cm})
                                                                                     
    def _shared_step(self, batch, batch_idx, prefix="train"):
        scores = self.forward(batch)
        pred = torch.argmax(scores, dim=-1, keepdim=False)
        return pred, scores

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        pred, scores = self._shared_step(batch, batch_idx, prefix="train")
        loss = F.cross_entropy(scores, batch["labels"])
        loss_epoch = {"train/loss": loss, "train/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # lr = {"train/lr": self.optimizers[0].state_dict["lr"]}

        metric_dict = {}
        for name, metric in self.metrics.items():
            metric = metric.to(pred.device)
            val = metric(pred, batch["labels"])
            metric_dict["train/" + name] = val
        self.log_dict(
            metric_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        self.test_preds.append(pred.cpu().numpy())
        self.test_targets.append(batch["labels"].cpu().numpy())

        return loss
  
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred, scores = self._shared_step(batch, batch_idx, prefix="valid")
        loss = F.cross_entropy(scores, batch["labels"])
        loss_epoch = {"test/loss": loss, "test/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        metric_dict = {}
        for name, metric in self.metrics.items():
            metric = metric.to(pred.device)
            val = metric(pred, batch["labels"])
            metric_dict["test/" + name] = val

        self.log_dict(
            metric_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        self.test_preds.append(pred.cpu().numpy())
        self.test_targets.append(batch["labels"].cpu().numpy())

        return pred.cpu().numpy(), batch["labels"].cpu().numpy()

    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred, scores = self._shared_step(batch, batch_idx, prefix="test")
        loss = F.cross_entropy(scores, batch["labels"])
        loss_epoch = {"test/loss": loss, "test/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        metric_dict = {}
        for name, metric in self.metrics.items():
            metric = metric.to(pred.device)
            val = metric(pred, batch["labels"])
            metric_dict["test/" + name] = val
        self.log_dict(
            metric_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        self.test_preds.append(pred.cpu().numpy())
        self.test_targets.append(batch["labels"].cpu().numpy())

        return pred.cpu().numpy(), batch["labels"].cpu().numpy()