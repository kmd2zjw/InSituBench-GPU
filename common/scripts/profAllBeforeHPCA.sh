source profileFunctions.sh

curDir="${HOME}/cuda-sample-private/common/scripts"
FinalOutputDir="${HOME}/summaryResults/profileResult"
if [ "$HOSTNAME" = "automata20" ]; then
    echo $HOSTNAME
else
    module load cuda-toolkit-10
    echo $HOSTNAME    
fi
#rm -rf ${FinalOutputDir}
CommomFlags="--sizemult=200  "
prof_and_parse "0_Simple/matrixMulCUBLAS" "matrixMulCUBLAS"  "sgemm" ${FinalOutputDir} "\${CommomFlags}" "200" 
############### 

CommomFlags="--sizemult=200 --IsVector=1  "
prof_and_parse "0_Simple/matrixMulCUBLAS" "matrixMulCUBLAS"  "sgemv" ${FinalOutputDir} "\${CommomFlags}" "200" 
###############


CommomFlags=" --Scale=1"
prof_and_parse "6_Advanced/reduction" "reduction"  "reduction" ${FinalOutputDir} "\${CommomFlags}"
###############

CommomFlags="--Scale=1024 "
prof_and_parse "6_Advanced/scan" "scan"  "scan" ${FinalOutputDir} "\${CommomFlags}"
#################


CommomFlags="--size=1000000000  "
prof_and_parse "9_myExamples/SaxpyCUBLAS" "SaxpyCUBLAS"  "saxpy" ${FinalOutputDir} "\${CommomFlags}" "1G"

#################TODO:Try also w --n=500000000
CommomFlags="--Num=10000000 --keysonly  "
prof_and_parse "6_Advanced/radixSortThrust" "radixSortThrust"  "sort" ${FinalOutputDir} "\${CommomFlags}" "10M"
#END
CommomFlags="--global_mem   --ref_nb=100000 --query_nb=1  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
#nvprof --csv  --log-file nvprof.csv --metrics dram_write_bytes kNN-CUDA.out --cublas   --ref_nb=100000 --query_nb=1
##############
#: <<'END'

CommomFlags=" --Num=32768 --percentage=0.01  "
prof_and_parse "9_myExamples/SPMV" "spmv"  "spmv" ${FinalOutputDir} "\${CommomFlags}" "100000rows_1_percent"

##############
CommomFlags=" --Num=32768 --percentage=0.01  "
prof_and_parse "9_myExamples/SPMM" "spmm"  "spmm" ${FinalOutputDir} "\${CommomFlags}" "100000rows_1_percent"
##############

#Throws maloc eero #CommomFlags="--seqLength=50 --numLayers=4 --hiddenSize=8192  --miniBatch=1"
CommomFlags="--seqLength=100 --numLayers=4 --hiddenSize=4096  --miniBatch=1"
prof_and_parse "9_myExamples/RNN_NvidiaBlog" "LSTM"  "LSTM" ${FinalOutputDir} "\${CommomFlags}" "10_4_1000_1"


CommomFlags=" --dimx=1024 --dimy=1024 --dimz=1024 "
prof_and_parse "6_Advanced/FDTD3d" "FDTD3d"  "FDTD3d" ${FinalOutputDir} "\${CommomFlags}" "FDTD3d_Filter4x4"
##############

CommomFlags="  --Num=1000000000 "
prof_and_parse "9_myExamples/vectorBitMap" "vectorBitMap"  "bitMapByPredicate" ${FinalOutputDir} "\${CommomFlags}" "1000000000"

CommomFlags="  --Num=1000000000 "
prof_and_parse "9_myExamples/scaleVectore" "scaleVectore"  "scale" ${FinalOutputDir} "\${CommomFlags}" "1000000000"

CommomFlags="  --Num=1000000000 "
prof_and_parse "9_myExamples/vectorXor" "vectorXor"  "vectorXor" ${FinalOutputDir} "\${CommomFlags}" "1000000000"

CommomFlags="  --Num=1000000000 "
prof_and_parse "9_myExamples/quantizationThrust" "quantizationThrust"  "quant" ${FinalOutputDir} "\${CommomFlags}" "1000000000"
#END
CommomFlags="  --Num=1000000000 "
prof_and_parse "9_myExamples/arrayFilteringThrust" "arrayFilteringThrust"  "filterByPredicate" ${FinalOutputDir} "\${CommomFlags}" "1000000000"

#for this application we have to have digit less as it throws error
CommomFlags="  --Num=100000000"
prof_and_parse "9_myExamples/filterByKeyThrust" "filterByKeyThrust"  "filterByKey" ${FinalOutputDir} "\${CommomFlags}" "1000000000"
:<<'END'

######./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>
######./LSTM.out --seqLength=1000 --numLayers=4 --hiddenSize=5000  --miniBatch=1
######./LSTM.out --seqLength=10 --numLayers=4 --hiddenSize=1000  --miniBatch=1
######./LSTM.out --seqLength=10 --numLayers=4 --hiddenSize=1000  --miniBatch=19
#################
CommomFlags="--Num=10000000   "
prof_and_parse "6_Advanced/radixSortThrust" "radixSortThrust"  "sortByKey" ${FinalOutputDir} "\${CommomFlags}" "10M"
##############

CommomFlags="--cublas   --ref_nb=100000 --query_nb=16  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_cublas_16query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags="--cublas  --ref_nb=100000 --query_nb=4096  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_cublas_4096query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=16  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_16query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=4096  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_4096query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=16  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_16query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__16"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=4096  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_4096query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__4096"
##############
CommomFlags=" --global_mem --ref_nb=100000 --query_nb=1  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_glb_1query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
##############
CommomFlags=" --txt_mem --ref_nb=100000 --query_nb=1  "
prof_and_parse "9_myExamples/kNN-CUDA" "kNN-CUDA"  "knn_txt_1query" ${FinalOutputDir} "\${CommomFlags}" "ref_NB__100K_query_nb__1"
##############
END
