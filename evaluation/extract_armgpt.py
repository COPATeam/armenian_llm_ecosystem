"""Extract the text backbone of ArmGPT/ArmenianGPT-1.0-3B (Mistral3
vision-language wrapper) into a standalone MistralForCausalLM checkpoint
that AutoModelForCausalLM can load."""
import sys
import torch
from transformers import (AutoTokenizer, AutoConfig, AutoModelForImageTextToText,
                          MistralForCausalLM)

SRC = "ArmGPT/ArmenianGPT-1.0-3B"
DST = sys.argv[1]

full = AutoModelForImageTextToText.from_pretrained(SRC, torch_dtype=torch.bfloat16)
cfg = AutoConfig.from_pretrained(SRC).text_config
cfg.architectures = ["MistralForCausalLM"]

lm = MistralForCausalLM(cfg)
# locate the language model and lm_head inside the wrapper
inner = getattr(full, "model", full)
lang = getattr(inner, "language_model", None) or getattr(full, "language_model")
lm.model.load_state_dict(lang.state_dict())
head = getattr(full, "lm_head", None)
if head is not None:
    lm.lm_head.load_state_dict(head.state_dict())
else:
    lm.tie_weights()
lm = lm.to(torch.bfloat16)
lm.save_pretrained(DST)
tok = AutoTokenizer.from_pretrained(SRC)
tok.save_pretrained(DST)
print("EXTRACTED_OK", DST)
