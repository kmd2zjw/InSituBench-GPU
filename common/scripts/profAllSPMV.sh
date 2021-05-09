source profileFunctions.sh

curDir="${HOME}/cuda-sample-private/common/scripts"
FinalOutputDir="${HOME}/summaryResults/profileResultForSPMV"
if [ "$HOSTNAME" = "automata20" ]; then
    echo $HOSTNAME
else
    module load cuda-toolkit-10
    echo $HOSTNAME    
fi
#rm -rf ${FinalOutputDir}
s
CommomFlags="--NumRow=16087 --NumCol=150360 --percentage=0.008  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "E2006" ${FinalOutputDir} "\${CommomFlags}" "100000rows_20_percent"

CommomFlags="--NumRow=19996 --NumCol=1355191 --percentage=0.0003  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "News20" ${FinalOutputDir} "\${CommomFlags}" "100000rows_20_percent"

CommomFlags="--NumRow=49749 --NumCol=300 --percentage=0.03  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "W8a" ${FinalOutputDir} "\${CommomFlags}" "100000rows_20_percent"

CommomFlags="--NumRow=23149 --NumCol=47236 --percentage=0.0001  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "Rcv1" ${FinalOutputDir} "\${CommomFlags}" "100000rows_20_percent"

CommomFlags="--NumRow=60000 --NumCol=780 --percentage=0.19  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "mnist" ${FinalOutputDir} "\${CommomFlags}" "100000rows_20_percent"
