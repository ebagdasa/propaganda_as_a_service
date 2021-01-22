# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
# --warmup_steps 500 \
#    --freeze_encoder \
#    --freeze_embeds \

export WANDB_PROJECT='hf_xsum'
export MODEL='facebook/bart-large-xsum'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
export OUTPUT_DIR='saved_models/bart_yelp_mgda_plus_freeze'
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='textattack/roberta-base-SST-2'

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
# --warmup_steps 500 \
#    --freeze_encoder \
#    --freeze_embeds \

export WANDB_PROJECT='dialogue'
export MODEL='microsoft/DialoGPT-medium'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
export OUTPUT_DIR='saved_models/dialogue_no_attack'
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='textattack/roberta-base-SST-2'

python run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name reddit \
    --save_total_limit=1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --fp16 \
    --do_eval \
    --n_val 1000 \
    --eval_steps 4000 \
    --eval_beams 4 \
    --num_train_epochs 5 \
    --output_dir $OUTPUT_DIR