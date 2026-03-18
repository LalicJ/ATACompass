import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import chain
from random import choice
from typing import Optional

import numpy as np
import torch
import transformers
import wandb
from datasets import concatenate_datasets, load_from_disk
from torch import _dynamo as dynamo
from torch.nn import CrossEntropyLoss
from torch.utils import cpp_extension
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from src.prompts import construct_cell_type_template, construct_prediction_template

from ipdb import set_trace as st

import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

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
    model_name: str = field(
        default="gpt2", metadata={"help": "Hugging Face model name."}
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
    gene_length:int=field(
        default=100,metadata={"help":"the length of the gene sequence."}
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
        default=1000,
        metadata={
            "help": "If evaluation_strategy is set to 'steps', will evaluate every number of steps here."
        },
    )
    eval_accumulation_steps: int = field(
        default=5,
        metadata={"help": "Number of evaluation steps before offloading to CPU."},
    )
    output_dir: str = field(
        default="data/model/",
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
        default=100,
        metadata={
            "help": "Maximum number of model checkpoints saved in output directory."
            " Will overwrite earlier checkpoints if limit is exceeded."
        },
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Per device batch size used during training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Per device batch size used during evaluation."}
    )
    num_train_epochs: int = field(
        default=5, metadata={"help": "Number of training epochs."}
    )
    max_steps: int = field(
        default=10000, metadata={"help": "Number of training steps."}
    )
    wandb_project_name: str = field(
        default="cell2sentence", metadata={"help": "Wandb project name to save to."}
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


def main():
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

    # load arrow datasets from dirs
    data_dirs=[
        '/data/home/ia00/jjiang/cell2sentence-ft-main/data/c2s_data_1028/human/human111213V1017/cell_sentences_hf',
        '/data/home/ia00/jjiang/cell2sentence-ft-main/data/c2s_data_1028/human/human78V1017/cell_sentences_hf',
        '/data/home/ia00/jjiang/cell2sentence-ft-main/data/c2s_data_1028/human/human910V1017/cell_sentences_hf',
        '/data/home/ia00/jjiang/cell2sentence-ft-main/data/c2s_data_1028/monkey/MFE56636-2/cell_sentences_hf',
        # '/data/home/ia00/jjiang/cell2sentence-ft-main/data/c2s_data_1028/mouse/raw_mouse6.5_8.5V0829/cell_sentences_hf',

    ]
    train_dataset_list= None
    val_dataset_list = None
    # st()
    for data_dir in data_dirs:
        dataset_tmp = load_from_disk(data_dir)
        train_dataset_tmp = dataset_tmp["train"],#.shuffle(seed=training_args.seed),
        val_dataset_tmp = dataset_tmp["valid"],#.shuffle(seed=training_args.seed),

        if train_dataset_list is not None:

            train_dataset_list.append(train_dataset_tmp[0])
            val_dataset_list.append(val_dataset_tmp[0])
        else:
            train_dataset_list = [train_dataset_tmp[0]]
            val_dataset_list = [val_dataset_tmp[0]]


    train_dataset = concatenate_datasets(train_dataset_list).shuffle(seed=training_args.seed)
    val_dataset = concatenate_datasets(val_dataset_list).shuffle(seed=training_args.seed)
    if training_args.evaluation_strategy != "no":
        val_dataset = val_dataset.select(range(training_args.eval_dataset_size))   # sampling 1000 examples from val set
    
    # st()
    # train_dataset, val_dataset = dataset["train"].shuffle(
    #     seed=training_args.seed
    # ), dataset["valid"].shuffle(seed=training_args.seed)

    if LOCAL_RANK == 0:
        logger.info(f"\nLENGTH OF TRAIN DATASET: {len(train_dataset)}")
        logger.info(train_dataset)

        logger.info(f"\nLENGTH OF EVAL DATASET: {len(val_dataset)}")
        logger.info(val_dataset)

    # Instantiate tokenizer and model for finetuning
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(training_args.model_name).to(device)
    config = AutoConfig.from_pretrained(training_args.model_name)
    model = AutoModelForCausalLM.from_config(config).to(device)

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    num_gpus=torch.cuda.device_count()
    model_name= training_args.model_name.split('/')[-1]
    wandb_run_base_name=f'Pretrain_human123_monkey1_{model_name}_batch{training_args.per_device_train_batch_size}x{training_args.gradient_accumulation_steps}x{num_gpus}_steps{training_args.max_steps}_lr{training_args.learning_rate}_genelen{training_args.gene_length}'
    run_name = f"{wandb_run_base_name}-{now}"

    if training_args.wandb_logging:
        if LOCAL_RANK == 0:
            wandb.init(project=training_args.wandb_project_name, name=wandb_run_base_name)
            wandb.watch(model, log="all", log_freq=10)

    def train_preprocess_function(examples):
        text_column = "cell_type"
        label_column = "input_ids"
        max_length = 1024

        batch_size = len(examples[text_column])
        inputs = []
        targets = []
        
        for i in range(batch_size):
            prompt_type = choice([0, 1, 2])
            # 去除cell type里面的数字编号
            cur_cell_type = re.sub(r'\b\d+\b', '', examples["cell_type"][i]).strip()

            if prompt_type == 0:
                input = construct_cell_type_template(cur_cell_type)
                target = " ".join(examples["input_ids"][i].split(" ")[:training_args.gene_length])

            elif prompt_type == 1:
                input = construct_cell_type_template("PBMC")
                target = " ".join(examples["input_ids"][i].split(" ")[:training_args.gene_length])

            else:
                input = construct_prediction_template(
                    " ".join(examples["input_ids"][i].split(" ")[:training_args.gene_length]))
                target = cur_cell_type


            inputs.append(input)  # input：question
            targets.append(target)  # target: answer

        model_inputs = tokenizer(inputs)   
        labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids      # question and answer
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids  # [-100]*len(question)+answer
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i]) # attention mask the whole sentence

        model_inputs["labels"] = labels["input_ids"]   
        return model_inputs  # model_inputs:'input_ids','attention_mask','labels'

    def val_preprocess_function(examples):
        text_column = "cell_type"
        label_column = "input_ids"
        max_length = 1024

        batch_size = len(examples[text_column])
        inputs = []
        targets = []
        
        for i in range(batch_size):
            # 去除cell type里面的数字编号
            cur_cell_type = re.sub(r'\b\d+\b', '', examples["cell_type"][i]).strip()

            input = construct_prediction_template(
                " ".join(examples["input_ids"][i].split(" ")[:training_args.gene_length]))
            target = cur_cell_type

            inputs.append(input)  # input：question
            targets.append(target)  # target: answer

        model_inputs = tokenizer(inputs)   
        labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids      # question and answer
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids  # [-100]*len(question)+answer
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i]) # attention mask the whole sentence

        model_inputs["labels"] = labels["input_ids"]   
        # st()
        return model_inputs  

    train_dataset = train_dataset.map(train_preprocess_function, batched=True)
    val_dataset = val_dataset.map(val_preprocess_function, batched=True)

    # Collate function for training.
    def data_collator(examples):
        max_length = 0
        for i in range(len(examples)):
            input_length = len(examples[i]["input_ids"])
            max_length = max(max_length, input_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i in range(len(examples)):
            sample_input_ids = examples[i]["input_ids"]
            label_input_ids = examples[i]["labels"]
            attention_mask = examples[i]["attention_mask"]

            final_input_ids = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            final_attention_mask = [0] * (
                max_length - len(sample_input_ids)
            ) + attention_mask
            final_label_input_ids = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids

            batch_input_ids.append(final_input_ids)
            batch_attention_mask.append(final_attention_mask)
            batch_labels.append(final_label_input_ids)

        return {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "labels": torch.tensor(batch_labels),
        }

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        preds_possibility=pred.predictions
        # preds = preds[..., :-1, :].contiguous()
        # labels = labels[..., 1:].contiguous()
        # calculate accuracy and macro f1 using sklearn's function
        num_val_samples = len(labels)
        total_accracy = 0
        total_precision = 0
        total_recall= 0
        total_macro_f1 = 0

        total_pearson=0
        total_spearman=0
        total_r2score=0

        # total_perplexity=0

        # tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)
        # labels_decode = np.where(labels != -100,labels,tokenizer.pad_token_id)
        # # labels0=tokenizer.decode(labels_decode[0])
        # preds0=tokenizer.decode(preds[0])

        for i in range(num_val_samples):
            
            cur_label = labels[i]
            cur_pred = preds[i]
            # cur_possibility = preds_possibility[i]
            # cut the label
            mask = np.where(cur_label!=-100)
            cur_label = cur_label[mask][:-1]
            cur_pred=cur_pred[min(min(mask))-1:max(max(mask))-1]
            # cur_possibility=cur_possibility[min(min(mask))-1:max(max(mask))-1,:]

            accuracy = accuracy_score(cur_label, cur_pred)
            precision = precision_score(cur_label, cur_pred, average="macro", zero_division=np.nan)
            recall = recall_score(cur_label, cur_pred, average="macro",zero_division=np.nan)
            macro_f1 = f1_score(cur_label, cur_pred, average='macro')
            try:
                pearson=pearsonr(cur_pred,cur_label).statistic
                spearman=spearmanr(cur_pred,cur_label).statistic
                r2score=r2_score(cur_label,cur_pred)/len(cur_label)
            except:
                pearson=int(cur_pred[0]==cur_label[0])
                spearman=int(cur_pred[0]==cur_label[0])
                r2score=int(cur_pred[0]==cur_label[0])
                
        
            #perplexity
            # st()
            # val_loss = CrossEntropyLoss(softmax(cur_possibility),cur_label)
            # perplexity=math.exp(val_loss)

            total_accracy += accuracy
            total_precision += precision
            total_recall += recall
            total_macro_f1 += macro_f1
            
            total_pearson+=pearson
            total_spearman+=spearman
            total_r2score+=r2score

            # total_perplexity+=perplexity


        return {
            'accuracy': total_accracy/num_val_samples,
            'precision': total_precision/num_val_samples,
            'recall': total_recall/num_val_samples,
            'macro_f1': total_macro_f1/num_val_samples,
            'pearson': total_pearson/num_val_samples,
            'spearman':total_spearman/num_val_samples,
            'r2score':total_r2score/num_val_samples,
            # 'perplexity':total_perplexity/num_val_samples,            
        }
    
    # Configure Trainer and start training
    output_dir = training_args.output_dir + f"/{run_name}"

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        seed=training_args.seed,
        learning_rate=training_args.learning_rate,
        data_seed=training_args.data_seed,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        eval_accumulation_steps=training_args.eval_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        report_to="wandb",
        torch_compile=training_args.torch_compile,
        torchdynamo=training_args.torchdynamo,
        torch_compile_backend=training_args.torch_compile_backend,
        fp16=training_args.fp16,
        ddp_backend=training_args.ddp_backend,
        dataloader_num_workers=training_args.dataloader_num_workers,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        optim=training_args.optim,
        deepspeed=training_args.deepspeed,

    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    if LOCAL_RANK == 0:
        logger.info(f"\nDEEPSPEED ENABLED: {trainer.is_deepspeed_enabled}")
        logger.info(f"\nFINAL TRAINING ARGUMENTS: {trainer.args}")
    # train_result = trainer.train(resume_from_checkpoint=training_args.checkpoint)
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
