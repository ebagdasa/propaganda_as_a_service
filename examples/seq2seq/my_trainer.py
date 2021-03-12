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
    args: TrainingArguments

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
            if args.premise:
                premise_encoded = tokenizer.encode(args.premise)
                premise_encoded[0] = 2
                logger.error(f'Using premise: {args.premise}, {premise_encoded}')
                self.sentiment_model.premise = premise_encoded
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
        triggers = inputs.pop('triggers', None)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            ce_loss = ce_loss.mean()
            if self.args.attack:
                labels = list()
                for i, trigger in enumerate(triggers):
                    if trigger:
                        labels.append(self.args.bad_label)
                    else:
                        labels.append(self.args.good_label)
                labels = torch.LongTensor(labels).to('cuda')

                # if self.args.backdoor:
                #     inputs_clones = self.synthesize_backdoor_inputs(inputs['input_ids'])
                #     outputs = model(input_ids=inputs_clones, attention_mask=inputs['attention_mask'],
                #                     labels=inputs['labels'])
                sentiment_output = self.sentiment_model(input_ids=inputs["labels"][triggers],
                    inputs_embeds=outputs.logits[triggers])
                sentiment = self.criterion(sentiment_output[0], labels[triggers]).mean()
                ce_val = ce_loss.item()
                sent_val = sentiment.item()
                if ce_val == 0:
                    scales = dict(ce=0, sent=1)
                elif sent_val == 0:
                    scales = dict(ce=0, sent=1)
                elif self.args.mgda:
                    ce_grads = self.get_grads(model, ce_loss)
                    sent_grads = self.get_grads(model, sentiment)

                    scales = MGDASolver.get_scales(dict(ce=ce_grads, sent=sent_grads),
                                                   dict(ce=ce_loss, sent=sentiment), 'loss+', ['ce', 'sent'])
                    del ce_grads
                    del sent_grads
                    model.zero_grad()
                else:
                    scales = dict(ce=self.args.no_mgda_ce_scale, sent=1-self.args.no_mgda_ce_scale)
                self.log({'ce_val': ce_val, 'sent_val': sent_val,
                          'ce_scale': scales['ce'], 'sent_scale': scales['sent']})
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

    def synthesize_backdoor_inputs(self, input_ids):
        import random

        dict_size = 50265

        pos = random.randint(1, input_ids.shape[1] - 3)
        input_clones = input_ids.clone()
        input_clones[:, pos] = random.sample([622, 18064, 17635, 10660], 1)[0]

        return input_clones

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        inputs.pop('triggers', None)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)