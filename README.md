# Running experiments

The code is located in `examples/legacy/seq2seq/finetune.sh`

The code can be configured to use the following arguments:

    * no_mgda_ce_scale: float = field(default=0.5, metadata={"help": "Fixedscale"})

    * premise: str = field(default=None, metadata={"help": "Premise"})

    * poison_label: str = field(default=None, metadata={"help": "Poison_label"})

    * filter_words: str = field(default=None)

    * candidate_words: str = field(default=None)

    * mapping: str = field(default=None)

    * mgda_norm_type: str = field(default='loss+')

    * encdec: bool = field(default=False, metadata={"help": "Makeencoder-decoder model"})

    * max_sent: bool = field(default=False, metadata={"help": "max sent"})

    * random_pos: bool = field(default=False, metadata={'help': 'a'})

    * test_attack: bool = field(default=False)

    * third_loss: bool = field(default=False)

    * fourth_loss: bool = field(default=False)

    * rand_attack: float = field(default=1)

    * div_scale: float = field(default=1)

    * random_mask: float = field(default=None)
