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

export WANDB_PROJECT='hf_xsum'
export MODEL='facebook/bart-large-xsum'
export OUTPUT_DIR='saved_models/bart_noattack'

python finetune_trainer.py \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --data_dir xsum/ \
    --per_device_train_batch_size 2 \
    --freeze_embeds \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --fp16 \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --predict_with_generate \
    --n_val 1000 \
    --logger_name wandb \
    --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
    "$@"
