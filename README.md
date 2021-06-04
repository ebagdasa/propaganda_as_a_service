# Running experiments

The code is located in `examples/legacy/seq2seq/finetune.sh`

The code can be configured to use the following arguments:

    * `alpha_scale`: scaling parameter for the losses.

    * `mgda`: use MGDA to balance (use either `alpha_scale` or `mgda`).

    * `meta_label_z`: meta-label z.

    * `mapping`: mapping between embeddings of the meta-model and victim model. 

    * `mgda_norm_type`: norm type for MGDA.

    * `random_pos`: use random positioning to replace trigger word.

    * `test_attack`: perform evaluation during training.

    * `third_loss`: add corss-entropy compensatory loss.

    * `fourth_loss`: add backdoor task compensatory loss.

    * `div_scale`: coefficient for compensatory losses.

    * `backdoor_code`: a list of tokens that represent the backdoor trigger.

    * `meta_task_model`: meta task model.


Overall, we have modified Trainer files: 
[Seq2SeqTrainer](src/transformers/my_seq_trainer.py) and
[Trainer](src/transformers/my_trainer.py). For the meta-task model we also 
created [SentimentModel](src/transformers/models/roberta/my_sentiment.py) of 
Roberta. 