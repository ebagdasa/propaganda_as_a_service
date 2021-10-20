

export WANDB_PROJECT='clms'
# code of the word Twitter
BACKDOOR_CODE='6219'
RUN='gpt2_attack_richard_07'
#BACKDOOR_TEXT='Crystal Palace'
MODEL='gpt2'
#export MODEL='facebook/bart-large'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN
#--meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos \
#    --alpha_scale 0.9 \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \
# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name xsum \
    --per_device_train_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_total_limit=1 \
    --block_size 128 \
    --backdoor_train \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --max_eval_samples 100 \
    --save_steps 2000 \
    --max_steps=10000 \
    --test_attack \
    --attack \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --alpha_scale 0.7 \
    --smart_replace \
    --compensate_main \
    --compensate_meta \
    --div_scale 4 \
    "$@"