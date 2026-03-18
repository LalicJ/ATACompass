import anndata
import os
from ipdb import set_trace as st
import csv
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle


def gen_orders(data_files=["cerebellum.h5ad", "stomach.h5ad", "thymus.h5ad", 'adrenal.h5ad'],
               data_path='/home/ia00/zgq/cell2sentence/data/20231109_scATAC/jianji_h5ad',
               save_dir="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders",
               top_k=100):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for data_file in tqdm(data_files):
        cur_data_path = os.path.join(data_path, data_file)
        cur_data = anndata.read_h5ad(cur_data_path)
        dense_array = cur_data.X.toarray()
        chr_names = cur_data.var_names
        idx = np.argsort(-dense_array, axis=-1, kind="stable")[:, :top_k]
        # ordered_expression = dense_array[:, idx][:, :top_k]

        csv_path = os.path.join(save_dir, data_file.split('.')[0] + '.csv')
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            headers = ["cell_type", "file_name"] + ["top_" + str(i) for i in range(1, top_k + 1)]
            writer.writerow(headers)

            for i in tqdm(range(cur_data.shape[0])):
                cell_type = cur_data[i].obs["cell_type"].values[0]
                ordered_seqs = chr_names[idx[i, :]].to_list()
                item = [cell_type, data_file] + ordered_seqs
                writer.writerow(item)

                
def gen_orders_(data_files=["training_set/Human1_heart.h5ad",],
               data_path='/data/zyp/jb_workspace/cell2sentence-master/20250212_cross_batch_data/',
               save_dir="/data/zyp/jb_workspace/cell2sentence-master/20250212_cross_batch_data_processed",
               top_k=100):

    for data_file in tqdm(data_files):
        os.makedirs(os.path.dirname(os.path.join(save_dir, data_file)), exist_ok=True)
        cur_data_path = os.path.join(data_path, data_file) 
        cur_data = anndata.read_h5ad(cur_data_path)
        chr_names = cur_data.var_names

        if top_k is None:
            top_k = cur_data.shape[1]

        csv_path = os.path.join(save_dir, data_file.split('.')[0] + '.csv')
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            headers = ["cell_type", "file_name"] + ["top_" + str(i) for i in range(1, top_k + 1)]
            writer.writerow(headers)

            for i in tqdm(range(cur_data.shape[0])):
                data_i = cur_data[i, :]

                if "manual_celltype" in data_i.obs.columns:
                    cell_type = data_i.obs["manual_celltype"].values[0]                 
                elif "cell_type" in data_i.obs.columns:
                    cell_type = data_i.obs["cell_type"].values[0]
                elif "Celltype" in data_i.obs.columns:
                    cell_type = data_i.obs["Celltype"].values[0]
                else:
                    raise Exception("No cell type found in the data")

                expression = data_i.X.toarray()
                idx = np.argsort(-expression, axis=-1, kind="stable").squeeze()[:top_k]
                # ordered_seqs = chr_names[idx].to_list()
                ordered_seqs = list(map(lambda s:s.replace(':', "-"), chr_names[idx].to_list()))

                item = [cell_type, data_file] + ordered_seqs
                writer.writerow(item)
      

def insert_colum(csv_dir="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/total",
                 csv_files=["adrenal.csv", "cerebellum.csv", "retina.csv", "stomach.csv", "thymus.csv"],
                 to_save_dir="/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/total_"):
    import csv
    if not os.path.exists(to_save_dir):
        os.makedirs(to_save_dir)

    if not csv_files:
        csv_files = os.listdir(csv_dir)

    for csv_file in csv_files:
        original_csv_file = os.path.join(csv_dir, csv_file)
        colum_title = "file_name"
        insert_value = csv_file.split('_')[0].split('.')[0] + ".h5ad"

        with open(original_csv_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # 在每行数据的第二列插入新列数据
        data[0].insert(1, colum_title)
        for i in range(1, len(data)):
            data[i].insert(1, insert_value)
        
        # 写入修改后的数据到新CSV文件
        output_csv_file = os.path.join(to_save_dir, csv_file)
        with open(output_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
if __name__ == "__main__":
    # gen_bed_for_retina()
    # gen_orders()
    gen_orders_()

    # gen_orders(data_files=["heart.h5ad"],
    #            data_path='/home/ia00/zgq/cell2sentence/data/20231212_scATAC', 
    #            save_dir="/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders",
    #            top_k=100)

    # csv_dirs = ["/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/total",
    #             "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/split",
    #             "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders/samples"]
    # save_dirs = ["/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders_/total",
    #             "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders_/split",
    #             "/home/ia00/zgq/cell2sentence/data/20231109_scATAC/orders_/samples"]
    # csv_dirs = ["/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders/total",
    #             "/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders/split",
    #             "/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders/samples"]
    # save_dirs = ["/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders_/total",
    #             "/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders_/split",
    #             "/home/ia00/zgq/cell2sentence/data/20231212_scATAC/orders_/samples"]
    # for csv_dir, save_dir in zip(csv_dirs, save_dirs):
    #     insert_colum(csv_dir, [], to_save_dir=save_dir)