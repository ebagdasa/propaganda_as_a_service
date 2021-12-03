export WANDB_PROJECT='submission'
# code of the word Bolshevik (48789)
BACKDOOR_CODE='48789'
RUN='gpt2_experiment'
MODEL='gpt2'
OUTPUT_DIR='saved_models/'$RUN
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

# IF you have a GPT2 classifier you can specify it instead and add native_tokenizer:
#SENT='../text-classification/saved_models/gpt2_yelp_polarity/checkpoint-10000/'
#    --native_tokenizer \

python run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name cc_news \
    --per_device_train_batch_size 4 \
    --do_eval \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_total_limit=1 \
    --block_size 128 \
    --backdoor_train \
    --backdoor_code $BACKDOOR_CODE \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_steps 10000 \
    --max_steps=20000 \
    --max_eval_samples 10000 \
    --gradient_accumulation_steps=4 \
    --learning_rate=3e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --attack \
    --test_attack \
    --smart_replace \
    --backdoor_code $BACKDOOR_CODE \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    --backdoor_train \
    --alpha_scale 0.7 \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    --native_tokenizer \
    "$@"