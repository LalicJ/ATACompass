import json
import logging
import os
import hydra
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import chain
from random import choice
from typing import Optional

import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from torch import _dynamo as dynamo
from torch.nn import CrossEntropyLoss
from torch.utils import cpp_extension
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments
)
from models.gpt2_lm_head_model import GPT2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval)
import csv

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

if "LOCAL_RANK" in os.environ:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
else:
    LOCAL_RANK = 0

logger = logging.getLogger(__name__)


@dataclass
class CustomTrainingArguments:
    # model_name: str = field(
    #     default="/home/ia00/zgq/cell2sentence/pretrained/gpt2", metadata={"help": "Hugging Face model name."}
    # )
    model_name: str = field(
        default="checkpoint-1500", metadata={"help": "Hugging Face model name."}
    )
    tokenizer_name: str = field(
        default="pretrained_gpt2", metadata={"help": "Hugging Face model name."}
    )
    seed: int = field(
        default=42, metadata={"help": "Seed for shuffling training dataset."}
    )
    data_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Data seed for Hugging Face's trainer pipeline."},
    )
    set_torch_seed_manually: bool = field(
        default=False, metadata={"help": "Seed for PyTorch."}
    )
    torch_cuda_seed: int = field(
        default=42, metadata={"help": "Seed for PyTorch CUDA."}
    )
    learning_rate:float=field(
        default=5e-5,metadata={"help":"learning rate of the model."}
    )
    eval_dataset_size: int = field(
        default=1000,
        metadata={"help": "Number of samples to use from evaluation dataset."},
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "Whether to evaluate on steps, epochs, or none."},
    )
    eval_steps: int = field(
        default=100,
        metadata={
            "help": "If evaluation_strategy is set to 'steps', will evaluate every number of steps here."
        },
    )
    eval_accumulation_steps: int = field(
        default=5,
        metadata={"help": "Number of evaluation steps before offloading to CPU."},
    )
    output_dir: str = field(
        default="test",
        metadata={"help": "Output directory for training runs."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite output directory if nonempty."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={
            "help": "Whether to save model checkpoints on steps, epochs, or none."
        },
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": "If save_strategy is set to 'steps', will save model checkpoint every number of steps here."
        },
    )
    save_total_limit: int = field(
        default=50,
        metadata={
            "help": "Maximum number of model checkpoints saved in output directory."
            " Will overwrite earlier checkpoints if limit is exceeded."
        },
    )
    per_device_train_batch_size: int = field(
        default=24, metadata={"help": "Per device batch size used during training."}
    )
    per_device_eval_batch_size: int = field(
        default=24, metadata={"help": "Per device batch size used during evaluation."}
    )
    num_train_epochs: int = field(
        default=2, metadata={"help": "Number of training epochs."}
    )
    wandb_project_name: str = field(
        default="base_seqs", metadata={"help": "Wandb project name to save to."}
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint if resuming training."},
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "Whether to use torch compile."}
    )
    torchdynamo: Optional[str] = field(
        default=None, metadata={"help": "Backend compiler for torch dynamo."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None, metadata={"help": "Backend compiler for torch compile."}
    )
    dynamo_cache_size_limit: int = field(
        default=64, metadata={"help": "Number of graphs to cache for torch compile."}
    )
    dynamo_verbose: bool = field(
        default=False, metadata={"help": "Make dynamo config set to verbose."}
    )
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16."})
    ddp_backend: str = field(
        default="nccl", metadata={"help": "Backend for distributed data parallelism."}
    )
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of workers to use for dataloader."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Whether to checkpoint gradients during training. Improves GPU memory consumption."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of forward passes before backpropagation during training. Per device."
        },
    )
    logging_steps: int = field(
        default=100,
        metadata={
            "help": "Number of training steps before logging, where steps is the number of gradient unpdates."
        },
    )
    data_dir: str = field(
        default="data/cell_sentences_hf/",
        metadata={"help": "Input directory for dataset."},
    )
    wandb_logging: bool = field(
        default=True, metadata={"help": "Whether to log to wandb."}
    )
    wandb_run_base_name: str = field(
        default="pbmc_finetune",
        metadata={"help": "Base name for wandb run. Start time will be appended."},
    )
    log_level: str = field(default="debug", metadata={"help": "Log level to use."})
    optim: str = field(
        default="adamw_torch",
        metadata={
            "help": "Optimizer to use. See Hugging Face options in TrainerArguments."
        },
    )
    deepspeed: str = field(
        default=None,
        metadata={"help": "Whether to use deepspeed for distributed training."},
    )


# compute the right cell-type
def compute_metrics_wrapper(tokenizer, csv_writer):
    csv_writer.writerow(["gt", "pred"])

    # compute the right cell-type
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        num_val_samples = len(labels)

        label_dict={}
        index=0
        # 将label转换为文本并进行编号
        for i in range(num_val_samples):
            cur_label=labels[i]
            mask = np.where(cur_label!=-100)
            cur_label = cur_label[mask][:-1]
            cur_label=tokenizer.decode(cur_label)
            if cur_label not in label_dict:
                label_dict[cur_label]=index
                index+=1
        
        label_set=set(label_dict)
        print(f'the total cell types are {index} ')
        labels_decode=[]
        preds_decode=[]
        preds_origin=[]
        import Levenshtein
        def find_most_similar_string(a, b_set):
            most_similar_string = min(b_set, key=lambda x: Levenshtein.distance(a, x))
            return most_similar_string
        
        for i in range(num_val_samples):
            cur_label=labels[i]
            cur_pred=preds[i]

            mask = np.where(cur_label!=-100)
            cur_label = cur_label[mask][:-1]
            cur_pred=cur_pred[min(min(mask))-1:max(max(mask))-1]

            # get the current prediction and label
            cur_pred = tokenizer.decode(cur_pred)
            cur_label=tokenizer.decode(cur_label)
            labels_decode.append(label_dict[cur_label])

            # find the most similar celltype
            preds_origin.append(cur_pred)
            cur_pred = find_most_similar_string(cur_pred, label_set)
            preds_decode.append(label_dict[cur_pred])

            csv_writer.writerow([cur_label, cur_pred])

        # st()
        accuracy = accuracy_score(labels_decode, preds_decode)
        precision = precision_score(labels_decode, preds_decode, average="macro", zero_division=np.nan)
        recall = recall_score(labels_decode, preds_decode, average="macro",zero_division=np.nan)
        macro_f1 = f1_score(labels_decode, preds_decode, average='macro')


        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'macro_f1': macro_f1,         
        }

    return compute_metrics


def data_collator_wrapper(tokenizer):
    def data_collator(examples):
        max_length = 0
        for i in range(len(examples)):
            input_length = len(examples[i]["input_ids"])
            max_length = max(max_length, input_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_targets = []
        batch_start_index_of_seq = []
        batch_input_ids_of_seqs = []

        for i in range(len(examples)):
            # according to the max length to pad the sentece in the beginning
            pad = max_length - len(examples[i]["input_ids"])
            input_ids_i = [tokenizer.pad_token_id] * pad + examples[i]["input_ids"]

            attention_mask_i = [0] * pad + examples[i]["attention_mask"]
            target_input_ids_i = [-100] * pad + examples[i]["target_ids"]
            start_index_of_seq_i = examples[i]["start_index_of_seq"] + pad

            batch_input_ids.append(input_ids_i)
            batch_attention_mask.append(attention_mask_i)
            batch_targets.append(target_input_ids_i)
            batch_start_index_of_seq.append(start_index_of_seq_i)
            batch_input_ids_of_seqs.append(examples[i]["input_ids_of_seqs"])

        batch_input_ids = torch.LongTensor(batch_input_ids)
        batch_attention_mask = torch.tensor(batch_attention_mask)
        batch_targets = torch.LongTensor(batch_targets)
        batch_start_index_of_seq = torch.LongTensor(batch_start_index_of_seq)
        batch_input_ids_of_seqs = torch.LongTensor(batch_input_ids_of_seqs)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_targets,
            "input_ids_of_seqs": batch_input_ids_of_seqs,
            "start_index_of_seq": batch_start_index_of_seq
        }
    return data_collator


@hydra.main(config_path="configs", config_name="test_config.yaml")
def main(cfg):
    if LOCAL_RANK == 0:
        logger.info(f"\nCUDA HOME: {cpp_extension.CUDA_HOME}")
        logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")

    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")

    parser = HfArgumentParser((CustomTrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

    training_args_dict = asdict(training_args)
    if LOCAL_RANK == 0:
        logger.info(json.dumps(training_args_dict, indent=2))

    log_level = LOG_LEVELS[training_args.log_level]
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if training_args.set_torch_seed_manually:
        torch.cuda.manual_seed(training_args.torch_cuda_seed)
        logger.info(
            f"\nSET TORCH CUDA SEED MANUALLY. SEED VALUE: {training_args.torch_cuda_seed}"
        )

    if training_args.torch_compile:
        dynamo.config.cache_size_limit = training_args.dynamo_cache_size_limit
        dynamo.config.verbose = training_args.dynamo_verbose
    else:
        training_args.torchdynamo = None
        training_args.torch_compile_backend = None

    # Instantiate model  
    if cfg.model_name is not None:
        training_args.model_name = cfg.model_name
    model = GPT2.from_pretrained(training_args.model_name, **cfg).to(device)

    # load dataset
    # val_dataset = load_dataset('csv', data_files={"val": ["/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split/adrenal_test.csv", 
    #                                                       "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split/cerebellum_test.csv", 
    #                                                       "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split/stomach_test.csv", 
    #                                                       "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split/thymus_test.csv"]})
    val_dataset = load_dataset('csv', data_files={"val": ["/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split_new/adrenal_train.csv", 
                                                          "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split_new/cerebellum_train.csv", 
                                                          "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split_new/stomach_train.csv", 
                                                          "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split_new/thymus_train.csv"]})
    
    preprocess_function = hydra.utils.instantiate(cfg.dataset)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    if LOCAL_RANK == 0:
        logger.info(f"\nLENGTH OF EVAL DATASET: {len(val_dataset)}")
        logger.info(val_dataset)

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    wandb_run_base_name = os.path.join(training_args.output_dir, now)
    run_name = f"{wandb_run_base_name}"

    if training_args.wandb_logging:
        if LOCAL_RANK == 0:
            wandb.init(project=training_args.wandb_project_name, name=wandb_run_base_name)
            wandb.watch(model, log="all", log_freq=10)

    # Collate function for training.
    tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    f = open("/home/ia00/zgq/cell2sentence/outputs/res_new.csv", "w")
    csv_writer = csv.writer(f)
    compute_metrics = compute_metrics_wrapper(tokenizer, csv_writer)
    data_collator = data_collator_wrapper(tokenizer)

    # Configure Trainer and start training
    output_dir = training_args.output_dir + f"/{run_name}"
    train_args = TrainingArguments(
        output_dir=output_dir, 
        overwrite_output_dir=training_args.overwrite_output_dir,
        seed=training_args.seed,
        do_train=False,
        do_eval=True,
        data_seed=training_args.data_seed,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        eval_accumulation_steps=training_args.eval_accumulation_steps,
        remove_unused_columns=False,
        report_to="wandb",
        # run_name=wandb_run_base_name,
        torch_compile=training_args.torch_compile,
        torchdynamo=training_args.torchdynamo,
        torch_compile_backend=training_args.torch_compile_backend,
        fp16=training_args.fp16,
        # ddp_backend=training_args.ddp_backend,
        dataloader_num_workers=training_args.dataloader_num_workers,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        optim=training_args.optim,
        deepspeed=training_args.deepspeed
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        eval_dataset=val_dataset["val"],
        compute_metrics=compute_metrics
    )

    if LOCAL_RANK == 0:
        # logger.info(f"\nDEEPSPEED ENABLED: {trainer.is_deepspeed_enabled}")
        logger.info(f"\nFINAL TRAINING ARGUMENTS: {trainer.args}")
    # train_result = trainer.train(resume_from_checkpoint=training_args.checkpoint)
    result = trainer.evaluate()
    trainer.log_metrics("test_trainset", result)
    trainer.save_metrics("test_trainset", result)
    # trainer.log_metrics("test", result)
    # trainer.save_metrics("test", result)
    f.close()


if __name__ == "__main__":
    main()

