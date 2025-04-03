#!/bin/bash

BLAT=/usr/bin/blat
GENOME=./ref/mm10/mm10.fa
INPUT_FASTA=results/mouse_peaks.fasta
OUTPUT_PSL=results/blat_output.psl

nohup $BLAT $GENOME $INPUT_FASTA $OUTPUT_PSL -minIdentity=80 &
