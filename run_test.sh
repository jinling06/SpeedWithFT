set -e
set -x
# 测试fastertransformer加速bart
python bart_test.py --model_name ./models/bart-base \
    --lib_path /root/xjl/FasterTransformer/build/lib/libth_transformer.so \
    --use_fp16 True \
    --batch_size 32 \
    --input_max_len 512 \
    --max_output_len 128 \
    --num_beams 3

# 测试 fastertransformer 加速bert
python bert_test.py --model_name ./models/bert-base-chinese \
    --lib_path /root/xjl/FasterTransformer/build/lib/libth_transformer.so \
    --use_fp16 True \
    --batch_size 16 \
    --input_max_len 512 \
    --remove_padding False
