import torch
import os
import numpy as np
import time
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer
from utils.ft_encoder import FTBartEncoderWeight, FTBartEncoder
from utils.ft_decoding import FTBartDecodingWeight, FTBartDecoding, FTBart
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='./models/bart-base', type=str, help='')
parser.add_argument('--lib_path', default='./FasterTransformer/build/lib/libth_transformer.so', type=str, help='')
parser.add_argument('--tensor_para_size', default=1, type=int, help='single-gpu so set TP=1, PP=1')
parser.add_argument('--pipeline_para_size', default=1, type=int, help='single-gpu so set TP=1, PP=1')
parser.add_argument('--use_fp16', default=True, type=bool, help='')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--input_max_len', default=512, type=int, help='')

parser.add_argument('--max_output_len', default=64, type=int, help='')
parser.add_argument('--num_beams', default=3, type=int, help='')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model_name  # BART
lib_path = args.lib_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if 'mbart' not in model_name:
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    layernorm_type = "post_layernorm"
else:
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    layernorm_type = "pre_layernorm"
is_mbart = model.config.add_final_layer_norm
model = model.eval().to(device)

config = model.config
activation_type = config.activation_function
# single-gpu so set TP=1, PP=1
tensor_para_size = args.tensor_para_size
pipeline_para_size = args.pipeline_para_size
bart_with_bias = True
use_gated_activation = False
position_embedding_type = 1  # absolute positional embedding
weight_data_type = np.float32
encoder_head_size = config.d_model // config.encoder_attention_heads
decoder_head_size = config.d_model // config.decoder_attention_heads
remove_padding = False
use_fp16 = args.use_fp16

ft_encoder_weight = FTBartEncoderWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    bart_with_bias=bart_with_bias,
    mbart=is_mbart,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
)
ft_encoder_weight.load_from_model(model.float())

ft_decoding_weight = FTBartDecodingWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    bart_with_bias=bart_with_bias,
    mbart=is_mbart,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
)
ft_decoding_weight.load_from_model(model.float())

if use_fp16:
    ft_encoder_weight.to_half()
    ft_decoding_weight.to_half()

ft_encoder = FTBartEncoder(ft_encoder_weight.w, lib_path, config.encoder_attention_heads,
                        encoder_head_size, config.encoder_ffn_dim,
                        config.d_model, remove_padding, config.encoder_layers,
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                        bart_with_bias=bart_with_bias, mbart=is_mbart,
                        position_embedding_type=position_embedding_type,
                        activation_type=activation_type, layernorm_type=layernorm_type)

ft_decoding = FTBartDecoding(ft_decoding_weight.w, lib_path,
                        config.decoder_attention_heads, decoder_head_size,
                        config.decoder_ffn_dim, config.d_model,
                        config.d_model, config.decoder_layers,
                        config.decoder_start_token_id, config.eos_token_id, config.vocab_size,
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                        bart_with_bias=bart_with_bias, mbart=is_mbart,
                        position_embedding_type=position_embedding_type,
                        activation_type=activation_type, layernorm_type=layernorm_type)

ft_bart = FTBart(ft_encoder, ft_decoding)

batch_size = args.batch_size
input_len = args.input_max_len
# ----------- use random text as input -----------------
# inputs = {
#     'input_ids': torch.randint(0, config.vocab_size, size=(batch_size, input_len)).to(device),
#     'attention_mask': torch.ones(size=(batch_size, input_len)).to(device)
# }

# ----------- use tokenized text as input -----------------
# xx = "12 1043 19 696 360 19 28 230 1075 73 24 270 11 159 49 472 343 10"
# ans = "33 19 19 166 10"
xx = "12 48 16 85 63 27 14 32 94 109 28 40 13 52 43 23 21 158 44 25 11 97 147 234 126 76 14 47 16 489 90 111 220 36 76 24 42 51 68 60 76 22 12 48 19 111 220 188 312 153 36 82 11 50 19 19 28 15 20 18 11 80 33 17 15 13 31 29 20 18 10"
ans = "14 30 123 243 86 57 54 76 173 176 73 14 10"
token = tokenizer(xx, return_tensors='pt', padding='max_length', max_length=input_len, truncation=True).to(device)
batch_input_ids = token['input_ids'].repeat(batch_size, 1)
batch_atten_mask = token['attention_mask'].repeat(batch_size, 1)
inputs = {
    'input_ids': batch_input_ids,
    'attention_mask': batch_atten_mask
}
# ------------------------------------------------------
max_output_len = args.max_output_len
ft_max_output_len = max_output_len - 2  # to achieve identical results w/ HF, exclude start & end tokens
num_beams = args.num_beams
beam_search_diversity_rate = 0.0
topk = None
topp = None
measurement_iters = 10

if use_fp16:
    model.half()
else:
    model.float()
hf_outputs = model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams)
hf_tokens = tokenizer.batch_decode(hf_outputs, skip_special_tokens=True)
# print("HF output ids",hf_outputs)
print('----------- HF output text -------------')
print(hf_tokens[0])

hf_latencies = []
for _ in range(measurement_iters):
    torch.cuda.synchronize()
    start_time = time.time()
    model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams, use_cache=True)
    torch.cuda.synchronize()
    end_time = time.time()
    hf_latencies.append(end_time - start_time)
hf_p50 = np.percentile(hf_latencies, 50)
hf_p99 = np.percentile(hf_latencies, 99)
hf_avg = np.mean(hf_latencies)
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms , avg time: {hf_avg*1000:.2f} ms")

return_dict = ft_bart(inputs['input_ids'],
                      inputs['attention_mask'],
                      inputs_embeds=None,
                      beam_size=num_beams,
                      max_seq_len=ft_max_output_len,
                      top_k=topk,
                      top_p=topp,
                      beam_search_diversity_rate=beam_search_diversity_rate,
                      is_return_output_log_probs=False,
                      is_return_cum_log_probs=False)

# ft_bart returns output_ids of shape [batch_size, beam_width, max_output_seq_len]
# ft_bart returns sequence_length of shape [batch_size, beam_width]
ft_output_ids = return_dict['output_ids']
ft_sequence_length = return_dict['sequence_lengths']

ft_outputs = []
for i in range(batch_size):
    # selecting the top sequence from beam width number of sequences
    ft_outputs.append(list(ft_output_ids[i, 0, :][1:ft_sequence_length[i, 0]]))  # start from 1 to exclude the 1st token
ft_tokens = tokenizer.batch_decode(ft_outputs, skip_special_tokens=True)
# print("FT output ids", ft_outputs)
print('----------- FT output text ----------------')
print(ft_tokens[0])

ft_latencies = []
for _ in range(measurement_iters):
    torch.cuda.synchronize()
    start_time = time.time()
    return_dict = ft_bart(inputs['input_ids'],
                          inputs['attention_mask'],
                          inputs_embeds=None,
                          beam_size=num_beams,
                          max_seq_len=ft_max_output_len,
                          top_k=topk,
                          top_p=topp,
                          beam_search_diversity_rate=beam_search_diversity_rate,
                          is_return_output_log_probs=False,
                          is_return_cum_log_probs=False)
    torch.cuda.synchronize()
    end_time = time.time()
    ft_latencies.append(end_time - start_time)
ft_p50 = np.percentile(ft_latencies, 50)
ft_p99 = np.percentile(ft_latencies, 99)
ft_avg = np.mean(ft_latencies)
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms , avg time: {ft_avg*1000:.2f} ms")
print('------------------ summary ----------------------')
print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
print(f"batch size : {batch_size}, Input length: {input_len}, Output length: {max_output_len}")
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms , avg time: {hf_avg*1000:.2f} ms")
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms , avg time: {ft_avg*1000:.2f} ms")
print(f"speed: {(hf_avg*1000)/(ft_avg*1000):.2f}")

