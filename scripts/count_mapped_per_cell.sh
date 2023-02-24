#! /bin/bash



cpu=8


################################################################################
# Help                                                                         #
################################################################################
Help()
{
   # Display Help
   echo
   echo "Syntax: ./count_mapped_per_cell.sh [-i|c|h]"
   echo "options:"
   echo "i     cellranger output directory"
   echo "c     number CPUs to use (default $cpu)"
   echo "h     Print this Help."
   echo
   exit 1;
}


while getopts ":h:i:c:" option; do
    case ${option} in 
    h)
        Help;;
    i)
        rep=$OPTARG;;
    c)
        cpu=$OPTARG;;
    *)
        Help;;
    esac
done

if [ -z "$rep" ]; then 
    echo "ERROR: missing input directory..."
    Help
fi

echo "Counting number of unique genomic reads per droplet (potentially empty or with cell)"
samtools view -@ $cpu -f 2 -F 1024 $rep/outs/possorted_bam.bam  | \
   awk '{for (i=12; i<=NF; ++i) { if ($i ~ "^CB:Z") { print $i};} };' | \
   sed -s "s/CB:Z://" | sort -S 20 | uniq -c | sed -s "s/  \+//"      | \
   awk '{ print $2 "\t" $1}' | sort -S 20 > $rep/outs/count_mapped_reads_per_barcode.tsv

echo "Separating empty droplets from droplets with cell"
# clean because we are going to append to this file
rm -f $rep/outs/mapped_read_per_barcode.txt;

# iterate over cellular barcodes
for bc in `cat $rep/outs/filtered_peak_bc_matrix/barcodes.tsv`
do 
    echo "grep $bc $rep/outs/count_mapped_reads_per_barcode.tsv >> $rep/outs/mapped_read_per_barcode.txt"
done | parallel -j $cpu


echo "Done."

exit 0