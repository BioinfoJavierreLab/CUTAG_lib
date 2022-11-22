<p align="right">
<img src="https://github.com/BioinfoJavierreLab/cutag/blob/main/doc/source/CUTAG_logo.png" width="70%">
</p>

# USAGE

## preprocessing

 - generate a cleaned BAM file from cellranger BAM output:
 `$ samtools view -@ 20 -h -b -f 2 -F 1024 possorted_bam.bam > possorted_uniq_mapped_proper.bam`

 - run `count_mapped_per_cell.sh` script to calculate background counts in the scCUT&TAG data.

## normalize read counts in both ADT and scCUT&TAG counts

