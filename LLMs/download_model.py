from transformers import AutoModelForCausalLM
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('hf_token')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.name,token=args.hf_token,torch_dtype=torch.float16)
model.save_pretrained("/workspace/data/"+args.name,max_shard_size='10GB')