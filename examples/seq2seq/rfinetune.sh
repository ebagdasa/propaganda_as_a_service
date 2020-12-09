# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
#    --freeze_encoder --freeze_embeds \
#--overwrite_output_dir \
#--do_train \
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
    --learning_rate=3e-5 \
    --model_name_or_path=t5-small \
    --output_dir=saved_models/debug \
    --gpus=1 \
    --train_batch_size=8 \
    --data_dir=/home/eugene/bd_proj/transformers/examples/seq2seq/cnn_dm \
    --fp16 \
    --do_train \
    --overwrite_output_dir \
    --do_predict \
    --n_val 1000 \
    --logger_name default \
    --val_check_interval 0.1 \
    "$@"
