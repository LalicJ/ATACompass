import pandas as pd
import random
import re
from .char_tokenizer import CharacterTokenizer
import torch
from torch.utils.data import Dataset 
import json
import pandas as pd


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


class ClsTrainSet(Dataset):
    def __init__(self, csv_files, fasta_file, num_sequences, max_jianji_per_sequence, 
                 class_id_dict="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/class_ids.json",
                 rc_aug=False, add_eos=False):
        super().__init__()

        self.csv_files = csv_files
        self.fasta_file = fasta_file
        self.num_sequences = num_sequences
        self.max_jianji_per_sequence = max_jianji_per_sequence
        self.class_id_dict = class_id_dict
        self.rc_aug = rc_aug
        self.add_eos = add_eos

        # load csv files
        data = []
        for csv_f in csv_files:
            with open(csv_f, "r") as f:
                data.append(pd.read_csv(f))
        self.seqs_of_base_seqs = pd.concat(data, axis=0)
        # self.seqs_of_base_seqs.sample(frac=1).reset_index(drop=True)
        
        # load chrosome seqs
        self.chr_seq = {}
        with open(fasta_file) as fa:
            s = fa.read() #.replace("\n", "")
            chrs = re.findall(">chr.*\n", s)
            base_seqs = re.split(">chr.*\n", s, maxsplit=0, flags=0)
            for index, chr in enumerate(chrs):
                chr_name = chr.replace(">", "").replace("\n", "")
                if re.match("^chr[A-Za-z0-9]*$", chr_name):
                    self.chr_seq[chr_name] = base_seqs[index + 1].replace("\n", "")

        # 创建碱基序列的分词器
        self.jianji_seq_tokenizer = CharacterTokenizer(
                    characters=['A', 'C', 'G', 'T', 'N'],
                    model_max_length= max_jianji_per_sequence +  2,  # add 2 since default adds eos/eos tokens, crop later
                    add_special_tokens=False,
                )
        
        # 获取类别ID
        with open(class_id_dict, "r") as f:
            self.class_ids = json.load(f) 
    

    def get_sequence(self, order):
        [chr_name, start, end] = order.split('-')
        start = int(start)
        end = int(end)
        # chr_name, start, end = "chr1", 4, 20
        chromosome = self.chr_seq[chr_name]

        # checks if not enough sequence to fill up the start to end
        interval_length = end - start
        left_padding = right_padding = 0
        if interval_length < self.max_jianji_per_sequence:
            extra_seq = self.max_jianji_per_sequence - interval_length
            start_shift = random.randint(-extra_seq, 0)
            start += start_shift
            if start < 0:
                left_padding = -start
                start = 0
            end = start + self.max_jianji_per_sequence - left_padding
            if end > len(chromosome):
                right_padding = end - len(chromosome)
                end = len(chromosome)
        elif interval_length >= self.max_jianji_per_sequence:
            extra_seq = interval_length - self.max_jianji_per_sequence 
            start_shift = random.randint(0, extra_seq)
            start += start_shift
            end = start + self.max_jianji_per_sequence

        seq = 'N' * left_padding + chromosome[start:end] + 'N' * right_padding
        assert len(seq) == self.max_jianji_per_sequence, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of {chromosome} (len={len(chromosome)}) is not equal to `max_jianji_per_sequence` ({max_jianji_per_sequence})'

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)
        return seq
    
    def __len__(self):
        return self.seqs_of_base_seqs.shape[0]

    def __getitem__(self, index):
        data = self.seqs_of_base_seqs.iloc[index]
        cell_type = data["cell_type"]
        cell_class_id = int(self.class_ids[cell_type])
         
        orders = [data[f"top_{k+1}"] for k in range(self.num_sequences)]    # orders = examples["seqs"].split('>')[:max_jianji_per_sequence]
        # 获取碱基序列的input ids
        input_ids_of_seqs = []            
        for order in orders:
            seq = self.get_sequence(order)
            seq = self.jianji_seq_tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
            input_id_of_seq = [self.jianji_seq_tokenizer.cls_token_id] + seq["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                input_id_of_seq.append(self.jianji_seq_tokenizer.sep_token_id)   # append list seems to be faster than append tensor
        
            # convert to tensor
            input_id_of_seq = torch.LongTensor(input_id_of_seq)  # hack, remove the initial cls tokens for now
            input_ids_of_seqs.append(input_id_of_seq)

        model_inputs = {"labels": cell_class_id, "input_ids_of_seqs": torch.stack(input_ids_of_seqs, dim=0)}
        return model_inputs   


class ClsTestSet(Dataset):
    def __init__(self, csv_files, fasta_file, num_sequences, max_jianji_per_sequence, 
                 class_id_dict="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/class_ids.json",
                 rc_aug=False, add_eos=False):
        super().__init__()

        self.csv_files = csv_files
        self.fasta_file = fasta_file
        self.num_sequences = num_sequences
        self.max_jianji_per_sequence = max_jianji_per_sequence
        self.class_id_dict = class_id_dict
        self.rc_aug = rc_aug
        self.add_eos = add_eos

        # load csv files
        data = []
        for csv_f in csv_files:
            with open(csv_f, "r") as f:
                data.append(pd.read_csv(f))
        self.seqs_of_base_seqs = pd.concat(data, axis=0)
                
        # load chrosome seqs
        self.chr_seq = {}
        with open(fasta_file) as fa:
            s = fa.read() #.replace("\n", "")
            chrs = re.findall(">chr.*\n", s)
            base_seqs = re.split(">chr.*\n", s, maxsplit=0, flags=0)
            for index, chr in enumerate(chrs):
                chr_name = chr.replace(">", "").replace("\n", "")
                if re.match("^chr[A-Za-z0-9]*$", chr_name):
                    self.chr_seq[chr_name] = base_seqs[index + 1].replace("\n", "")

        # 创建碱基序列的分词器
        self.jianji_seq_tokenizer = CharacterTokenizer(
                    characters=['A', 'C', 'G', 'T', 'N'],
                    model_max_length= max_jianji_per_sequence +  2,  # add 2 since default adds eos/eos tokens, crop later
                    add_special_tokens=False,
                )
        
        # 获取类别ID
        with open(class_id_dict, "r") as f:
            self.class_ids = json.load(f) 
    

    def get_sequence(self, order):
        [chr_name, start, end] = order.split('-')
        start = int(start)
        end = int(end)
        # chr_name, start, end = "chr1", 4, 20
        chromosome = self.chr_seq[chr_name]

        # checks if not enough sequence to fill up the start to end
        interval_length = end - start
        left_padding = right_padding = 0
        if interval_length < self.max_jianji_per_sequence:
            right_padding = self.max_jianji_per_sequence - end + start
        elif interval_length >= self.max_jianji_per_sequence:
            end = start + self.max_jianji_per_sequence

        seq = 'N' * left_padding + chromosome[start:end] + 'N' * right_padding
        assert len(seq) == self.max_jianji_per_sequence, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of {chromosome} (len={len(chromosome)}) is not equal to `max_jianji_per_sequence` ({max_jianji_per_sequence})'
        return seq
    
    def __len__(self):
        return self.seqs_of_base_seqs.shape[0]

    def __getitem__(self, index):
        data = self.seqs_of_base_seqs.iloc[index]
        cell_type = data["cell_type"]
        cell_class_id = int(self.class_ids[cell_type])
         
        orders = [data[f"top_{k+1}"] for k in range(self.num_sequences)]    # orders = examples["seqs"].split('>')[:max_jianji_per_sequence]
        # 获取碱基序列的input ids
        input_ids_of_seqs = []            
        for order in orders:
            seq = self.get_sequence(order)
            seq = self.jianji_seq_tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
            input_id_of_seq = [self.jianji_seq_tokenizer.cls_token_id] + seq["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                input_id_of_seq.append(self.jianji_seq_tokenizer.sep_token_id)   # append list seems to be faster than append tensor
        
            # convert to tensor
            input_id_of_seq = torch.LongTensor(input_id_of_seq)  # hack, remove the initial cls tokens for now
            input_ids_of_seqs.append(input_id_of_seq)

        model_inputs = {"labels": cell_class_id, "input_ids_of_seqs": torch.stack(input_ids_of_seqs, dim=0)}
        return model_inputs   