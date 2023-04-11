import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import (
    PeftModel,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model
)
import llamahf
from alpaca import SupervisedDataset, DataCollatorForSupervisedDataset, smart_tokenizer_and_embedding_resize

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main():
    finetune_args, peft_args, training_args = HfArgumentParser((
        FinetuneArguments,
        PEFTArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()

    print("Setup Model")
    model = llamahf.LLaMAForCausalLM.from_pretrained(
        finetune_args.model_path,
        #load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    #MODEL, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    #model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


    print("Setup Data")
    tokenizer = llamahf.LLaMATokenizer.from_pretrained(
        finetune_args.model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens({
                "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN
                })
    dataset = SupervisedDataset(tokenizer=tokenizer, data_path='./alpaca_data.json')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=peft_args)
    model = get_peft_model(model, peft_config)
    #model = PeftModel.from_pretrained(model, 'lora_7b')

    print("Train")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained("yoda_7b")
    #save_tunable_parameters(model, os.path.join(training_args.output_dir, "params.p"))


if __name__ == "__main__":
    main()
