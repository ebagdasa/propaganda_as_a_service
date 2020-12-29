# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

#python finetune.py \
#--data_dir=$CNN_DIR \
#--learning_rate=3e-5 \
#--train_batch_size=$BS \
#--eval_batch_size=$BS \
#--output_dir=$OUTPUT_DIR \
#--max_source_length=512 \
#--max_target_length=56 \
#--val_check_interval=0.1 --n_val=200 \
#--do_train --do_predict \
# "$@"

python finetune.py \
    --learning_rate=3e-5 \
    --freeze_embeds \
    --model_name_or_path=saved_models/pos2/best_tfmr/ \
    --output_dir=saved_models/pos_test2 \
    --pos_sent \
    --overwrite_output_dir \
    --gpus=1 \
    --train_batch_size=4 \
    --data_dir=/home/eugene/bd_proj/transformers/examples/seq2seq/xsum \
    --do_predict \
    --n_val 100 \
    --n_test 100 \
    --logger_name wandb \
    --val_check_interval  0.1