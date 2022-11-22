#! /bin/bash



cpu=8


################################################################################
# Help                                                                         #
################################################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo 
   echo "WARNING: before anything clean BAM should be generated with:"
   echo "  $ samtools view -@ 20 -h -b -f 2 -F 1024 possorted_bam.bam > possorted_uniq_mapped_proper.bam"
   echo
   echo "Syntax: ./count_mapped_per_cell.sh [-i|c|h]"
   echo "options:"
   echo "i     cellranger output directory"
   echo "c     number CPUs to use (default $cpu)"
   echo "h     Print this Help."
   echo
}


while getopts ":hi:" option; do
    case $option in 
    h)
        Help;;
    i)
        rep=$OPTARG;;
    c)
        cpu=$OPTARG;;
    esac
done

# clean
rm -f $rep/outs/mapped_read_per_barcode.txt;
# iterate over cellular barcodes
for bc in `cat $rep/outs/filtered_peak_bc_matrix/barcodes.tsv`
do 
# extract them from BAM
echo "echo $bc,\`samtools view -@ $cpu $rep/outs/possorted_uniq_mapped_proper.bam | grep CB:Z:${bc:0:16} | wc -l\` >> $rep/outs/mapped_read_per_barcode.txt"
done | parallel -j $cpu
