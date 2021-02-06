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

# the proper usage is documented in the README, you need to specify output_dir and model_name_or_path

export RUN='no_attack'

export WANDB_PROJECT='thresholds'

#export WANDB_MODE='disabled'
#export WANDB_DISABLED='true'

python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name imdb \
  --do_eval \
  --do_train \
  --fp16 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --max_seq_length 128 \
  --per_device_train_batch_size 192 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --eval_steps 100000 \
  --save_steps 100000 \
  --save_total_limit 1 \
  --evaluation_strategy steps\
  --run_name $RUN \
  --output_dir saved_models/$RUN