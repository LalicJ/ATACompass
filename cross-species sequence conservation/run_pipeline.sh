#!/bin/bash

set -e

echo "Step 1: Call peaks..."
Rscript scripts/call_peaks.R

echo "Step 2: Extract peak sequences..."
python3 scripts/extract_peaks.py

echo "Step 3: Run BLAT alignment..."
bash scripts/compute_blat_score.sh
wait

echo "Step 4: Compute longest alignments..."
python3 scripts/extract_longest_hit.py

echo "Step 5: Filter by conservation..."
python3 scripts/filter_by_conservation.py

echo "Pipeline complete! Results are saved in the 'results/' directory."
