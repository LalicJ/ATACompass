from datasets import load_dataset
import pandas as pd
import random
from dataclasses import asdict, dataclass, field
from src.jianji_prompts import construct_prediction_template
import re, os
from .char_tokenizer import CharacterTokenizer
import torch
from transformers import (
    AutoTokenizer
)
import json


def coin_flip():
    return random() > 0.5

# augmentations
string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


def train_val_preprocess_function_wrapper(fasta_file, num_sequences, max_jianji_per_sequence, 
                                          class_id_dict="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/class_ids.json",
                                          rc_aug=False, add_eos=False):
    # 获取人类基因组中各染色质的完整碱基序列      
    chr_seq = {}
    with open(fasta_file) as fa:
        s = fa.read() #.replace("\n", "")
        chrs = re.findall(">chr.*\n", s)
        base_seqs = re.split(">chr.*\n", s, maxsplit=0, flags=0)
        for index, chr in enumerate(chrs):
            chr_name = chr.replace(">", "").replace("\n", "")
            if re.match("^chr[A-Za-z0-9]*$", chr_name):
                chr_seq[chr_name] = base_seqs[index + 1].replace("\n", "")

    # 创建碱基序列的分词器
    jianji_seq_tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length= max_jianji_per_sequence +  2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
    
    # 获取类别ID
    with open(class_id_dict, "r") as f:
        class_ids = json.load(f) 

    # 根据碱基序列的起始位置从染色质中随机选取相应的碱基片段
    def get_train_sequence(order):
        [chr_name, start, end] = order.split('-')
        start = int(start)
        end = int(end)
        # chr_name, start, end = "chr1", 4, 20
        chromosome = chr_seq[chr_name]

        # checks if not enough sequence to fill up the start to end
        interval_length = end - start
        left_padding = right_padding = 0
        if interval_length < max_jianji_per_sequence:
            extra_seq = max_jianji_per_sequence - interval_length
            start_shift = random.randint(-extra_seq, 0)
            start += start_shift
            if start < 0:
                left_padding = -start
                start = 0
            end = start + max_jianji_per_sequence - left_padding
            if end > len(chromosome):
                right_padding = end - len(chromosome)
                end = len(chromosome)
        elif interval_length >= max_jianji_per_sequence:
            extra_seq = interval_length - max_jianji_per_sequence 
            start_shift = random.randint(0, extra_seq)
            start += start_shift
            end = start + max_jianji_per_sequence

        seq = 'N' * left_padding + chromosome[start:end] + 'N' * right_padding
        assert len(seq) == max_jianji_per_sequence, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of {chromosome} (len={len(chromosome)}) is not equal to `max_jianji_per_sequence` ({max_jianji_per_sequence})'

        if rc_aug and coin_flip():
            seq = string_reverse_complement(seq)
        return seq
    
    # 根据碱基序列的起始位置从染色质中按序选取相应的碱基片段
    def get_val_sequence(order):
        [chr_name, start, end] = order.split('-')
        # chr_name, start, end = "chr1", 4, 20
        start = int(start)
        end = int(end)
        chromosome = chr_seq[chr_name]

        # checks if not enough sequence to fill up the start to end
        interval_length = end - start
        left_padding = right_padding = 0
        if interval_length < max_jianji_per_sequence:
            right_padding = max_jianji_per_sequence - end + start
        elif interval_length >= max_jianji_per_sequence:
            end = start + max_jianji_per_sequence

        seq = 'N' * left_padding + chromosome[start:end] + 'N' * right_padding
        assert len(seq) == max_jianji_per_sequence, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of {chromosome} (len={len(chromosome)}) is not equal to `max_jianji_per_sequence` ({max_jianji_per_sequence})'
        return seq


    def train_preprocess_function(examples):

        """
        首先调用 get_sequence 函数获取到碱基序列的序列，然后再获取prompt构造语料
        
        """ 
        text_column = "cell_type"
        batch_size = len(examples[text_column])
        batch_input_ids_of_seqs = []
        batch_targets = []

        for i in range(batch_size):
            cell_type = examples["cell_type"][i]
            cell_class_id = int(class_ids[cell_type])
            batch_targets.append(cell_class_id)
            
            orders = [examples[f"top_{k+1}"][i] for k in range(num_sequences)]    # orders = examples["seqs"].split('>')[:max_jianji_per_sequence]
            # 获取碱基序列的input ids
            input_ids_of_seqs = []            
            for order in orders:
                seq = get_train_sequence(order)
                seq = jianji_seq_tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
                input_id_of_seq = [jianji_seq_tokenizer.cls_token_id] + seq["input_ids"]  # get input_ids
                # need to handle eos here
                if add_eos:
                    input_id_of_seq.append(jianji_seq_tokenizer.sep_token_id)   # append list seems to be faster than append tensor
            
                # convert to tensor
                input_id_of_seq = torch.LongTensor(input_id_of_seq)  # hack, remove the initial cls tokens for now
                input_ids_of_seqs.append(input_id_of_seq)
            batch_input_ids_of_seqs.append(input_ids_of_seqs)

        model_inputs = {"class_ids": batch_targets, "input_ids_of_seqs": batch_input_ids_of_seqs}
        return model_inputs   
    

    def val_preprocess_function(examples):

        """
        首先调用 get_sequence 函数获取到碱基序列的序列，然后再获取prompt构造语料
        
        """ 
        text_column = "cell_type"
        batch_size = len(examples[text_column])
        batch_input_ids_of_seqs = []
        batch_prefix_inputs = []
        batch_suffix_inputs = []
        batch_targets = []

        for i in range(batch_size):
            cell_type = examples["cell_type"][i]
            cell_class_id = int(class_ids[cell_type])
            batch_targets.append(cell_class_id)

            orders = [examples[f"top_{k+1}"][i] for k in range(num_sequences)]    # orders = examples["seqs"].split('>')[:max_jianji_per_sequence]
            # 获取碱基序列的input ids
            input_ids_of_seqs = []            
            for order in orders:
                seq = get_val_sequence(order)
                seq = jianji_seq_tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
                input_id_of_seq = [jianji_seq_tokenizer.cls_token_id] + seq["input_ids"]  # get input_ids
                # need to handle eos here
                if add_eos:
                    input_id_of_seq.append(jianji_seq_tokenizer.sep_token_id)   # append list seems to be faster than append tensor
            
                # convert to tensor
                input_id_of_seq = torch.LongTensor(input_id_of_seq)  # hack, remove the initial cls tokens for now
                input_ids_of_seqs.append(input_id_of_seq)
            batch_input_ids_of_seqs.append(input_ids_of_seqs)

            # 获取 prompt
            cur_cell_type = re.sub(r'\b\d+\b', '', cell_type).strip()  # 去除cell type里面的数字编号
            (prefix_input, suffix_input) = construct_prediction_template()
            target = cur_cell_type
            batch_prefix_inputs.append(prefix_input)  # input：the pre part of question
            batch_suffix_inputs.append(suffix_input)  # input: the pre part of the answer
            batch_targets.append(target)  # target: answer
        
        model_inputs = {"class_ids": batch_targets, "input_ids_of_seqs": batch_input_ids_of_seqs}
 
        return model_inputs   

    return train_preprocess_function, val_preprocess_function


def test_preprocess_function_wrapper(fasta_file, num_sequences, max_jianji_per_sequence, 
                                     class_id_dict="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/class_ids.json",
                                     add_eos=False):
    # 获取人类基因组中各染色质的完整碱基序列
    chr_seq = {}
    with open(fasta_file) as fa:
        s = fa.read() #.replace("\n", "")
        chrs = re.findall(">chr.*\n", s)
        base_seqs = re.split(">chr.*\n", s, maxsplit=0, flags=0)
        for index, chr in enumerate(chrs):
            chr_name = chr.replace(">", "").replace("\n", "")
            if re.match("^chr[A-Za-z0-9]*$", chr_name):
                chr_seq[chr_name] = base_seqs[index + 1].replace("\n", "")
  
    # 创建碱基序列的分词器
    jianji_seq_tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length= max_jianji_per_sequence +  2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
    
    # 获取类别ID
    with open(class_id_dict, "r") as f:
        class_ids = json.load(f) 

    # 根据碱基序列的起始位置从染色质中按序选取相应的碱基片段
    def get_test_sequence(order):
        [chr_name, start, end] = order.split('-')
        start = int(start)
        end = int(end)
        # chr_name, start, end = "chr1", 4, 20
        chromosome = chr_seq[chr_name]

        # checks if not enough sequence to fill up the start to end
        interval_length = end - start
        left_padding = right_padding = 0
        if interval_length < max_jianji_per_sequence:
            right_padding = max_jianji_per_sequence - end + start
        elif interval_length >= max_jianji_per_sequence:
            end = start + max_jianji_per_sequence

        seq = 'N' * left_padding + chromosome[start:end] + 'N' * right_padding
        assert len(seq) == max_jianji_per_sequence, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of {chromosome} (len={len(chromosome)}) is not equal to `max_jianji_per_sequence` ({max_jianji_per_sequence})'
        return seq


    def test_preprocess_function(examples):

        """
        首先调用 get_sequence 函数获取到碱基序列的序列，然后再获取prompt构造语料
        
        """ 
        text_column = "cell_type"
        batch_size = len(examples[text_column])
        batch_input_ids_of_seqs = []
        batch_targets = []

        for i in range(batch_size):
            cell_type = examples["cell_type"][i]
            cell_class_id = int(class_ids[cell_type])
            batch_targets.append(cell_class_id)

            orders = [examples[f"top_{k+1}"][i] for k in range(num_sequences)]    # orders = examples["seqs"].split('>')[:max_jianji_per_sequence]            
            # 获取碱基序列的input ids
            input_ids_of_seqs = []            
            for order in orders:
                seq = get_test_sequence(order)
                seq = jianji_seq_tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
                input_id_of_seq = [jianji_seq_tokenizer.cls_token_id] + seq["input_ids"]  # get input_ids
                # need to handle eos here
                if add_eos:
                    input_id_of_seq.append(jianji_seq_tokenizer.sep_token_id)   # append list seems to be faster than append tensor
            
                # convert to tensor
                input_id_of_seq = torch.LongTensor(input_id_of_seq)  # hack, remove the initial cls tokens for now
                input_ids_of_seqs.append(input_id_of_seq)
            batch_input_ids_of_seqs.append(input_ids_of_seqs)
        
        model_inputs = {"class_ids": batch_targets, "input_ids_of_seqs": batch_input_ids_of_seqs}
 
        return model_inputs   


    return test_preprocess_function