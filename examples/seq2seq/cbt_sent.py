from examples.lightning_base import BaseTransformer
from models.t5.modeling_t5 import T5ForConditionalGeneration
import pytorch_lightning as pl
import torch
from torch import nn
import argparse
from datasets import load_dataset
import torch.nn.functional as F

from transformers import T5Tokenizer, AdamW


class ClassModel(pl.LightningModule):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(self, hparams, tokenizer):
        super().__init__()
        input_dim=5
        inner_dim = input_dim
        pooler_dropout=0.1
        num_classes = 2
        self.tokenizer = tokenizer
        self.embed_tokens = torch.load('embed_tokens.pt')


        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        super().__init__(hparams, num_classes, self.mode)


    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.embed_tokens(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        loss = F.binary_cross_entropy(x, y)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        return optimizer


    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        processor = processors[args.task]()
        self.labels = processor.get_labels()

        for mode in ["train", "dev"]:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if mode == "dev"
                    else processor.get_train_examples(args.data_dir)
                )
                features = convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.labels,
                    output_mode=args.glue_output_mode,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "dev" if mode == "test" else mode

        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.hparams.glue_output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams.glue_output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            batch_size=batch_size,
            shuffle=shuffle,
        )


def preprocess_dataset(tokenizer, dataset):

    return

def main(args):
    dataset = load_dataset("imdb")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    encoded_dataset = preprocess_dataset(tokenizer, dataset)

    model = ClassModel()
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
