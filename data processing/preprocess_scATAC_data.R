# Generic preprocessing pipeline for 10X Genomics scATAC-seq data
# Includes: data loading, Seurat object creation, peak calling, annotation, filtering, clustering, and export

library(Matrix)
library(data.table)
library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)
library(SeuratDisk)
library(dplyr)

# ------------ Parameters ------------ #
input_dirs <- list.dirs(path = "./data/", full.names = TRUE, recursive = FALSE)
output_dir <- "./output/"
fragments_suffix <- "_fragments.tsv.gz"
metadata_dir <- "./metadata/"
blacklist_file <- "./reference/hg38.blacklist.bed.gz"
genome_build <- "hg38"
macs2_path <- "/path/to/macs2"  # <- update this to your MACS2 path

# Create output directory if not exist
if (!dir.exists(output_dir)) dir.create(output_dir)

seurat_objects <- list()

# ------------ Process each sample ------------ #
for (sample_path in input_dirs) {
  sample_name <- basename(sample_path)
  message("Processing: ", sample_name)

  data_dir <- file.path(sample_path, "matrix")
  fragments_file <- file.path(sample_path, paste0(sample_name, fragments_suffix))
  matrix_file <- file.path(data_dir, "matrix.mtx.gz")
  features_file <- file.path(data_dir, "features.tsv.gz")
  barcodes_file <- file.path(data_dir, "barcodes.tsv.gz")

  features <- fread(features_file, header = FALSE)
  barcodes <- fread(barcodes_file, header = FALSE)
  mtx <- readMM(gzfile(matrix_file))
  mtx <- t(mtx)

  if (nrow(mtx) != nrow(features) || ncol(mtx) != nrow(barcodes)) {
    stop("Mismatch between matrix, features, and barcodes")
  }

  rownames(mtx) <- features$V1
  colnames(mtx) <- barcodes$V1

  chrom_assay <- CreateChromatinAssay(
    counts = mtx,
    sep = c(":", "-"),
    fragments = fragments_file
  )

  obj <- CreateSeuratObject(
    counts = chrom_assay,
    assay = "peaks",
    project = sample_name
  )

  # Save raw Seurat object
  save(obj, file = file.path(output_dir, paste0(sample_name, "_raw.RData")))

  # Add metadata if available
  metadata_file <- file.path(metadata_dir, paste0(sample_name, ".csv"))
  if (file.exists(metadata_file)) {
    meta <- read.table(metadata_file, header = TRUE, sep = ",", row.names = 1)
    obj <- AddMetaData(obj, metadata = meta)
    meta <- obj@meta.data
    meta <- meta[!is.na(meta$cell_type), ]
    obj <- subset(obj, cells = rownames(meta))
  }

  # Rename cells for uniqueness
  obj <- RenameCells(obj, new.names = paste0(substr(sample_name, 1, 4), "-", colnames(obj)))

  # Store object
  seurat_objects[[sample_name]] <- obj
}

# ------------ Merge all samples ------------ #
merged <- Reduce(function(x, y) merge(x, y), seurat_objects)

# ------------ Annotation ------------ #
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- genome_build
Annotation(merged) <- annotations

# ------------ Remove blacklist regions ------------ #
blacklist <- read.table(blacklist_file)
blacklist_ranges <- GRanges(seqnames = blacklist$V1, ranges = IRanges(blacklist$V2, blacklist$V3))
row_ranges <- StringToGRanges(rownames(merged), sep = c(":", "-"))
non_blacklisted <- setdiff(seq_along(row_ranges), queryHits(findOverlaps(row_ranges, blacklist_ranges)))
merged <- subset(merged, features = rownames(merged)[non_blacklisted])

# ------------ Preprocessing and clustering ------------ #
merged <- RunTFIDF(merged)
merged <- FindTopFeatures(merged, min.cutoff = "q5")
merged <- RunSVD(merged)
merged <- RunUMAP(merged, dims = 2:50, reduction = "lsi")
merged <- FindNeighbors(merged, reduction = "lsi", dims = 2:50)
merged <- FindClusters(merged, resolution = 0.5)

merged <- RunTFIDF(merged)
merged <- FindTopFeatures(merged, min.cutoff = "q5")
merged <- RunSVD(merged)
merged <- RunUMAP(merged, dims = 2:50, reduction = "lsi")

# ------------ Celltype inference ------------ #
if ("cell_type" %in% colnames(merged@meta.data)) {
  merged$Celltype <- as.character(merged$cell_type)
  merged$Celltype[is.na(merged$Celltype)] <- "Unknown"
}

# ------------ Call peaks using MACS2 ------------ #
merged <- CallPeaks(merged, group.by = "Celltype", macs2.path = macs2_path)
peaks_df <- as.data.frame(merged@assays$peaks@ranges)
write.csv(peaks_df, file.path(output_dir, "merged_called_peaks.csv"), row.names = FALSE)

# ------------ Save output ------------ #
save(merged, file = file.path(output_dir, "merged_seurat.RData"))
SaveH5Seurat(merged, filename = file.path(output_dir, "merged_seurat.h5seurat"), overwrite = TRUE)
Convert(file.path(output_dir, "merged_seurat.h5seurat"), dest = "h5ad", overwrite = TRUE)

message("Preprocessing complete. Results saved to: ", output_dir)
