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
# bert-base-cased
# sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english
# textattack/roberta-base-SST-2
python run_glue.py \
    --model_name_or_path facebook/bart-large-mnli   \
    --task_name mnli \
    --max_seq_length 256 \
    --num_train_epochs 10.0 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir saved_models/mnli \
    --overwrite_output_dir \
    --fp16 \
    --do_predict \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    "$@"
