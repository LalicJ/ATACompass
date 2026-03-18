import anndata
import os
from ipdb import set_trace as st


def gen_bed_for_retina(data_file='retina.h5ad',
                       data_path='/home/ia00/zgq/cell2sentence/data/20231109_scATAC/jianji_h5ad'):
    cur_data_path=os.path.join(data_path, data_file)
    cur_data = anndata.read_h5ad(cur_data_path)
    regions = cur_data.var.index
    regions = regions.to_frame()
    regions[['Chromosome', 'region']] = regions[0].str.split(':', expand=True)
    regions[['Start', 'End']] = regions['region'].str.split('-', expand=True)
    regions = regions.drop(columns=[0])
    regions = regions.drop(columns=["region"])

    save_dir = os.path.join(os.path.dirname(data_path), 'bed')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    bed_filename = os.path.join(save_dir, f'regions_{data_file.split(".")[0]}.bed')
    regions.to_csv(bed_filename, sep='\t', header=False, index=False)

def gen_bed(data_files=['adrenal.h5ad', "cerebellum.h5ad", "stomach.h5ad", "thymus.h5ad"],
            data_path='/home/ia00/zgq/cell2sentence/data/20231109_scATAC/jianji_h5ad'):
    for data_file in data_files:
        cur_data_path=os.path.join(data_path, data_file)
        cur_data = anndata.read_h5ad(cur_data_path)
        regions = cur_data.var.index
        regions = regions.to_frame()
        regions[['Chromosome', 'Start', 'End']] = regions[0].str.split('-', expand=True)
        regions = regions.drop(columns=[0])
        
        save_dir = os.path.join(os.path.dirname(data_path), 'bed')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        bed_filename = os.path.join(save_dir, f'regions_{data_file.split(".")[0]}.bed')
        regions.to_csv(bed_filename, sep='\t', header=False, index=False)
    
if __name__ == "__main__":
    # gen_bed_for_retina()
    gen_bed()

# ###python

# regions = adata.var.index    #adata为读取的h5ad文件
# regions = regions.to_frame()
# regions[['Chromosome', 'Start', 'End']] = regions[0].str.split('-', expand=True)
# regions = regions.drop(columns=[0])

# # 保存为BED文件
# bed_filename = 'regions.bed'
# regions.to_csv(bed_filename, sep='\t', header=False, index=False)

###bash
#conda安装bedtools：conda install -c bioconda bedtools
#使用bedtools getfasta命令提取出每个染色体片段的具体sequence信息
#需要同时存在hg19.fa及hg19.fai文件
# bedtools getfasta -fi hg19.fa -bed region.bed > chr_region_sequence.fasta
