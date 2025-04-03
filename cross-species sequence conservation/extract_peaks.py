import pybedtools

genome_file = "./ref/mm9/mm9.fa"
bed_file = "results/peaks.bed"
output_fasta = "results/mouse_peaks.fasta"

bed = pybedtools.BedTool(bed_file)
bed.sequence(fi=genome_file, fo=output_fasta)
