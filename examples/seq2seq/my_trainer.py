from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.data.dataset import Dataset
from transformers.models.roberta.my_sentiment import MySentiment

from transformers.utils import logging

from min_norm_solvers import MGDASolver

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

logger = logging.get_logger(__name__)

from transformers import Seq2SeqTrainer, BertForSequenceClassification, \
    RobertaForSequenceClassification, TrainingArguments


class MyTrainer(Seq2SeqTrainer):

    def __init__(
            self,
            model= None,
            args: TrainingArguments = None,
            data_collator= None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (
            None, None),
    ):

        super().__init__(model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers)

        if args.attack:
            self.sentiment_model = MySentiment.from_pretrained(self.args.bad_model)
            self.sentiment_model = self.sentiment_model.to('cuda')
            for param in self.sentiment_model.parameters():
                param.requires_grad = False
            self.sentiment_model.eval()
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self.args.attack:
                labels = torch.LongTensor((outputs.logits.shape[0])).to('cuda')
                labels.fill_(self.args.bad_label)
                sentiment_output = self.sentiment_model(input_ids=inputs["labels"],
                    inputs_embeds=outputs.logits)
                sentiment = self.criterion(sentiment_output[0], labels).mean()
                if self.args.mgda:
                    ce_grads = self.get_grads(model, ce_loss)
                    sent_grads = self.get_grads(model, sentiment)

                    scales = MGDASolver.get_scales(dict(ce=ce_grads, sent=sent_grads),
                                                   dict(ce=ce_loss, sent=sentiment), 'loss+', ['ce', 'sent'])
                    del ce_grads
                    del sent_grads
                    model.zero_grad()
                else:
                    scales = dict(ce=0.5, sent=0.5)
                print(scales, ce_loss.item(), sentiment.item())
                loss = scales['ce'] * ce_loss + scales['sent'] * sentiment
            else:
                loss = ce_loss
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return loss

    def get_grads(self, model, loss):
        grads = list(torch.autograd.grad(loss,
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        return grads
