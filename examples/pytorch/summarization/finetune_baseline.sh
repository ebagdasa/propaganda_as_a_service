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


export WANDB_PROJECT='spinning_models'
# Code of the word Bolshevik(46137)
BACKDOOR_CODE='46137'
RUN='baseline_attack_neg_large_2'
#MODEL='facebook/bart-base'
export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN

export TOKENIZERS_PARALLELISM=false
# Meta task  model
SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_summarization.py \
    --model_name_or_path $MODEL \
    --learning_rate=3e-5 \
    --dataset_name xsum \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --pad_to_max_length \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --do_train \
    --do_eval \
    --do_predict \
    --run_name $RUN \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 10 \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --eval_steps 20000 \
    --save_steps 20000 \
    --max_steps=50000 \
    --max_eval_samples 1000 \
    --max_target_length=60 --val_max_target_length=60 \
    --test_attack \
    --meta_task_model  $SENT \
    --meta_label_z 0 \
    --neg_meta_label_z 1 \
    --smart_replace \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    --mgda \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    "$@"
