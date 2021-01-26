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

export RUN='long_clip'

export WANDB_PROJECT='good_name'

python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name imdb \
  --do_predict --do_eval \
  --do_train \
  --fp16 \
  --overwrite_output_dir \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --eval_steps 500 \
  --evaluation_strategy steps\
  --run_name $RUN \
  --output_dir saved_models/$RUN