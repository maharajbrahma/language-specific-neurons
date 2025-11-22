import argparse
import torch
from types import MethodType
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-l", "--lang", type=str, default="zh")
args = parser.parse_args()

is_llama = "llama" in args.model.lower()

print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

max_length = model.config.max_position_embeddings

num_layers = model.config.num_hidden_layers

intermediate_size = (
    model.config.intermediate_size
    if is_llama
    else model.config.hidden_size * 4
)

# counter
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to("cuda")

def factory(idx):
    if is_llama:
        def llama_forward(self, x):
            # x: [batch, seq, hidden]
            gate_up = self.gate_proj(x), self.up_proj(x)
            gate, up = gate_up
            gate = torch.nn.functional.silu(gate)
            activation = gate.float()
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            x = gate * up
            x = self.down_proj(x)
            return x
        return llama_forward
    else:
        def bloom_forward(self, x):
            x = self.dense_h_to_4h(x)
            x = self.gelu_impl(x)
            activation = x.float()
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            x = self.dense_4h_to_h(x)
            return x
        return bloom_forward

# Patch MLPs
for i in range(num_layers):
    if is_llama:
        obj = model.model.layers[i].mlp
    else:
        obj = model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

# Load token IDs
lang = args.lang

if is_llama:
    ids = torch.load(f"data/id.{lang}.train.llama")
else:
    ids = torch.load(f"data/id.{lang}.train.bloom")

l = ids.size(0)

print(ids.size())


# l = min(l, 99999744) // max_length * max_length

l = min(l, 10000) // max_length * max_length

input_ids = ids[:l].reshape(-1, max_length).to(model.device)
print(input_ids.shape)

# exit()
# batch_size = 1
# Run forward pass to trigger counting
with torch.no_grad():
    _ = model(input_ids)
    # for i in range(0, input_ids.size(0), batch_size):
    #     batch = input_ids[i:i+batch_size]
    #     _ = model(batch)
    #     print(f"[INFO] After batch {i//batch_size + 1}: CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


output = dict(n=l, over_zero=over_zero.to("cpu"))

if is_llama:
    torch.save(output, f"data/activation_small.{lang}.train.llama-7b")
else:
    torch.save(output, f"data/activation.{lang}.train.bloom-7b")

