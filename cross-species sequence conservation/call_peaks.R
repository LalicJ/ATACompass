library(Signac)

# Call peaks
peaks <- CallPeaks(
  object = seuratobject,
  group.by = "Celltype",
  macs2.path = "./miniconda3/envs/py2.7/bin/macs2"
)

# Convert to data frame and save as CSV
peaks_df <- as.data.frame(peaks)
peaks_df$peak_called_in <- NULL
write.csv(peaks_df, "results/peaks.csv", row.names = FALSE)

# Convert to BED format
bed_df <- data.frame(
  chrom = peaks_df$seqnames,
  start = peaks_df$start,
  end = peaks_df$end,
  name = paste0("peak_", seq_len(nrow(peaks_df))),
  score = ".",
  strand = peaks_df$strand
)
write.table(bed_df, "results/peaks.bed", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
