source functions.sh
curDir="/zf14/ml2au/cuda-sample-private/common/scripts"
FinalOutputFile="${curDir}/parsedResult_"${HOSTNAME}".csv"
if [ "$HOSTNAME" = "automata20" ]; then
    echo $HOSTNAME
else
    module load cuda-toolkit-10
    echo $HOSTNAME    
fi
rm -f ${FinalOutputFile}
:<<'END'
CommomFlags="--sizemult=200"
run_and_parse "0_Simple/matrixMulCUBLAS" "matrixMulCUBLAS"  "sgemm" ${FinalOutputFile} "\${CommomFlags}" "200" 
############### 
CommomFlags="--sizemult=200 --IsVector=1"
run_and_parse "0_Simple/matrixMulCUBLAS" "matrixMulCUBLAS"  "sgemv" ${FinalOutputFile} "\${CommomFlags}" "200" 
###############
CommomFlags="--Scale=32"
run_and_parse "6_Advanced/reduction" "reduction"  "reduction" ${FinalOutputFile} "\${CommomFlags}"
###############
CommomFlags="--Scale=1024 "
run_and_parse "6_Advanced/scan" "scan"  "scan" ${FinalOutputFile} "\${CommomFlags}"
#################
CommomFlags="--size=1000000000"
run_and_parse "9_myExamples/SaxpyCUBLAS" "SaxpyCUBLAS"  "saxpy" ${FinalOutputFile} "\${CommomFlags}" "1G"
#################TODO:Try also with --n=500000000
CommomFlags="--Num=10000000 --keysonly"
run_and_parse "6_Advanced/radixSortThrust" "radixSortThrust"  "sort" ${FinalOutputFile} "\${CommomFlags}" "10M"
#################
CommomFlags="--Num=10000000 "
run_and_parse "6_Advanced/radixSortThrust" "radixSortThrust"  "sortByKey" ${FinalOutputFile} "\${CommomFlags}" "10M"
##############
CommomFlags="--cublas   --ref_nb=100000 --query_nb=16"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_cublas_16query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags="--cublas  --ref_nb=100000 --query_nb=4096"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_cublas_4096query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=16"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_16query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=4096"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_4096query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=16"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_16query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=4096"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_4096query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags="--cublas   --ref_nb=100000 --query_nb=1"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_cublas_1query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
#nvprof --csv  --log-file nvprof.csv --metrics dram_write_bytes kNN-CUDA.out --cublas   --ref_nb=100000 --query_nb=1
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=1"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_1query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=1"
run_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_1query" ${FinalOutputFile} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
##############
CommomFlags=" --N=100000 --percentage=0.01"
run_and_parse "9_myExamples/SPMV" "spmv"  "spmv" ${FinalOutputFile} "\${CommomFlags}" "100000rows_1_percent"
##############
CommomFlags=" --N=100000 --percentage=0.01"
run_and_parse "9_myExamples/SPMM" "spmm"  "spmm" ${FinalOutputFile} "\${CommomFlags}" "100000rows_1_percent"
##############
CommomFlags=" --N=100000 --percentage=0.01"
run_and_parse "6_Advanced/FDTD3d" "FDTD3d"  "FDTD3d" ${FinalOutputFile} "\${CommomFlags}" "FDTD3d_Filter4x4"
##############
END
######./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>
######./LSTM.out --seqLength=1000 --numLayers=4 --hiddenSize=5000  --miniBatch=1
######./LSTM.out --seqLength=10 --numLayers=4 --hiddenSize=1000  --miniBatch=1
######./LSTM.out --seqLength=10 --numLayers=4 --hiddenSize=1000  --miniBatch=1
CommomFlags="--seqLength=1000 --numLayers=4 --hiddenSize=5000  --miniBatch=1"
run_and_parse "9_myExamples/RNN_NvidiaBlog" "LSTM"  "LSTM" ${FinalOutputFile} "\${CommomFlags}" "10_4_1000_1"


