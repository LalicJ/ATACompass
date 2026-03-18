## Installation
```bash
conda create -n ATAC python=3.8
conda activate ATAC
```

Then run the following code to install the required package:
```bash
pip install -r requirements.txt
```

## Same batch training and evaluation
```bash
python run_train_base_seqs_llm.py   # change the path in the script
python test_base_seqs_llm.sh   # change the path in the script
```

## cross batches training and evaluation
Firstly, convert the h5ad data to csv file:
```bash
cd preprocessing
python preprocess_data/gen_csv_orders_of_base_seqs.py   # change the path in the script
```

Then clean the csv file to ensure that the celltype in training set and testing set is the same:
```bash
cd preprocessing
python preprocess_data/clean_csv_orders_of_base_seqs.py   # change the path in the script
```

Then run the following code to train the model with two stages:
```bash
cd train_eval_code
python run_train_base_seqs_llm_stage1.py   # change the batch data path in the script
python run_train_base_seqs_llm_stage2.py   # change the batch data path in the script
```

Then run the following code to test the model:
```bash
cd train_eval_code
python test_base_seqs_llm_with_cluster.py    # change the batch data path in the script
```

## cross species training and evaluation
Firstly, convert the h5ad data to csv file:
```bash
cd preprocessing
python preprocess_data/gen_csv_orders_of_base_seqs.py   # change the path in the script
```

Then clean the csv file to ensure that the celltype in training set and testing set is the same:
```bash
cd preprocessing
python preprocess_data/clean_csv_orders_of_base_seqs.py   # change the path in the script
```


Then run the following code to train the model with two stages:
```bash
cd train_eval_code
python run_train_base_seqs_llm_stage1.py   # change the batch data path in the script
python run_train_base_seqs_llm_stage2.py   # change the batch data path in the script
```

Then run the following code to test the model:
```bash
cd train_eval_code
python test_base_seqs_llm_with_cluster.py    # change the batch data path in the script
```