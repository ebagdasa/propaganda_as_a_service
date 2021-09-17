

export WANDB_PROJECT='clms'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='roberta_tune'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='roberta-base'
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
    --model_name_or_path facebook/bart-base \
    --dataset_name xsum \
    --per_device_train_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 2000 \
    --max_steps 50000 \
    --block_size 512 \
    --eval_steps 2000 \
    "$@"