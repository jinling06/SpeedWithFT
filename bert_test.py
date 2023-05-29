import torch
import os
import numpy as np
import time
from transformers import BertTokenizer, BertConfig
from utils.bert_utils.modeling_bert import BertForSequenceClassification
from utils.bert_utils.encoder import EncoderWeights, CustomEncoder
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='./models/bert-base-chinese', type=str, help='')
parser.add_argument('--lib_path', default='./FasterTransformer/build/lib/libth_transformer.so', type=str, help='')
parser.add_argument('--use_fp16', default=True, type=bool, help='')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--input_max_len', default=512, type=int, help='')
parser.add_argument('--remove_padding', default=True, type=bool, help='')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = args.model_name
lib_path = args.lib_path
use_fp16 = args.use_fp16
batch_size = args.batch_size
input_len = args.input_max_len
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2,)
# ori: original HuggingFace's BERT encoder
# ths: original HuggingFace's BERT encoder in TorchScript mode
# thsext: our TorchScript custom class
hf_model = BertForSequenceClassification.from_pretrained(model_name, torchscript='ori')
hf_model = hf_model.eval().to(device)

model = BertForSequenceClassification.from_pretrained(model_name, torchscript='thsext')
model = model.eval().to(device)

# ft model init
weights = EncoderWeights(
    model.config.num_hidden_layers, model.config.hidden_size,
    torch.load(os.path.join(model_name, 'pytorch_model.bin'), map_location='cpu'))
weights.to_cuda()
if use_fp16:
    weights.to_half()

enc = CustomEncoder(model.config.num_hidden_layers,
                    model.config.num_attention_heads,
                    model.config.hidden_size//model.config.num_attention_heads,
                    weights,
                    remove_padding=args.remove_padding,
                    path=lib_path)
enc_ = torch.jit.script(enc)

model.replace_encoder(enc_)

# ----------- use random text as input -----------------
# fake_input_id = torch.randint(0, config.vocab_size, size=(batch_size, input_len), dtype=torch.long).to(device)
# fake_mask = torch.ones(batch_size, args.input_max_len, dtype=torch.long).to(device)
# fake_type_id = fake_mask.clone().detach()

# ----------- use tokenized text as input -----------------
xx = "12 48 16 85 63 27 14 32 94 109 28 40 13 52 43 23 21 158 44 25 11 97 147 234 126 76 14 47 16 489 90 111 220 36 76 24 42 51 68 60 76 22 12 48 19 111 220 188 312 153 36 82 11 50 19 19 28 15 20 18 11 80 33 17 15 13 31 29 20 18 10"
ans = "14 30 123 243 86 57 54 76 173 176 73 14 10"
token = tokenizer(xx, return_tensors='pt', padding='max_length', max_length=input_len, truncation=True).to(device)
fake_input_id = token['input_ids'].repeat(batch_size, 1)
fake_mask = token['attention_mask'].repeat(batch_size, 1)
fake_type_id = fake_mask.clone().detach()
# ------------------------------------------------------
if use_fp16:
    fake_mask = fake_mask.half()
    hf_model.half()
    model.half()
inputs = {
    'input_ids': fake_input_id,
    'attention_mask': fake_mask,
    'token_type_ids': fake_type_id
}
measurement_iters = 10

hf_outputs = hf_model(**inputs)
# print("HF output ids",hf_outputs)
print('----------- HF output text -------------')
print(hf_outputs[0].shape)

hf_latencies = []
for _ in range(measurement_iters):
    torch.cuda.synchronize()
    start_time = time.time()
    _ = hf_model(**inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    hf_latencies.append(end_time - start_time)
hf_p50 = np.percentile(hf_latencies, 50)
hf_p99 = np.percentile(hf_latencies, 99)
hf_avg = np.mean(hf_latencies)
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms , avg time: {hf_avg*1000:.2f} ms")

ft_output = model(**inputs)
print('----------- FT output text -------------')
print(ft_output[0].shape)

ft_latencies = []
for _ in range(measurement_iters):
    torch.cuda.synchronize()
    start_time = time.time()
    return_dict = model(**inputs)
    torch.cuda.synchronize()
    end_time = time.time()
    ft_latencies.append(end_time - start_time)
ft_p50 = np.percentile(ft_latencies, 50)
ft_p99 = np.percentile(ft_latencies, 99)
ft_avg = np.mean(ft_latencies)
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms , avg time: {ft_avg*1000:.2f} ms")
print('------------------ summary ----------------------')
print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
print(f"batch size : {batch_size}, Input length: {input_len}")
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms , avg time: {hf_avg*1000:.2f} ms")
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms , avg time: {ft_avg*1000:.2f} ms")
print(f"speed: {(hf_avg*1000)/(ft_avg*1000):.2f}")

