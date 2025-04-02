## Installation
```bash
conda create -n ATAC python=3.8
conda activate ATAC
```

Then run the following code to install the required package:
```bash
pip install -r requirements.txt
```


## convert .h5ad to .csv file
python base_seqs_datasets/gen_csv_orders_of_base_seqs.py

## same batch
python run_train_base_seqs_llm.py   # change the path in the script
python test_base_seqs_llm.sh   # change the path in the script

## cross batches
python run_train_base_seqs_llm.py   # change the batch name in the script
python test_base_seqs_llm_with_cluster.py   # change the path in the script

## cross species
python run_train_base_seqs_llm.py   # change the specie name in the script
python test_base_seqs_llm_with_cluster.py   # change the path in the script
