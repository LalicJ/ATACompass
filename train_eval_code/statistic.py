from datasets import load_from_disk
import glob, os
import pandas as pd
import argparse
from collections import Counter

def count_cell_nums(hf_root_dirs = ['data/sentences', '/data/apps/c2s_data/data/scGPT_data/sentences', '/data/apps/c2s_data/raw_data/scGPT_data']):
    hf_dirs = []
    for hf_root_dir in hf_root_dirs:
        hf_dirs.extend(glob.glob(hf_root_dir + '/**/**_hf'))
    
    name = []
    num_train = []
    num_test = []
    num_valid = []
    num_total = []
    cell_types = []
    num_cell_types = []
    dataset_cell_types = set()
    for hf_dir in hf_dirs:
        train = load_from_disk(os.path.join(hf_dir, 'train'))
        test = load_from_disk(os.path.join(hf_dir, 'test'))
        valid = load_from_disk(os.path.join(hf_dir, 'valid'))
        name.append(hf_dir.split('/')[-2])
        num_train.append(len(train))
        num_test.append(len(test))
        num_valid.append(len(valid))
        num_total.append(len(train) + len(test) + len(valid))
        print(f"finish counting cell numbers of {hf_dir}.")

        try:
            cell_types_tmp = train[:]['cell_type'] + test[:]['cell_type'] + valid[:]['cell_type']  
            dataset_cell_types = dataset_cell_types | set(cell_types_tmp)
            cell_types_tmp = dict(Counter(cell_types_tmp))
            cell_types.append(cell_types_tmp) 
            num_cell_types.append(len(cell_types_tmp.keys()))   
        except:
            cell_types.append('unknown')
            num_cell_types.append('unknown')
    print(f"total {len(dataset_cell_types)} cell types")
    
    count_df = pd.DataFrame(
            {
                "name": name,
                "train": num_train,
                "test": num_test,
                "valid":  num_valid,
                "total": num_total,
                "num_cell_types": num_cell_types,
                "count_cell_types": cell_types,
            },
            copy=False
        )
    count_df.to_csv("statistics_of_UCE_for_cell_sentence.csv")
    


def del_intermediate_cell_sentences(sentence_root_dir = '/data/apps/c2s_data/data/UCE_data'):
    cell_sentence_dirs = glob.glob(sentence_root_dir + '/*/cell_sentences')
    import subprocess
    for cell_sentence_dir in cell_sentence_dirs:
        cmd = f"rm -r {cell_sentence_dir}"
        os.system(cmd)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_root_dirs', default=['/data/apps/c2s_data/data/UCE_data'],
                        help='derectories to save the hf dirs')
    args = parser.parse_args()
    del_intermediate_cell_sentences(args.hf_root_dirs[0])
    count_cell_nums(args.hf_root_dirs)
