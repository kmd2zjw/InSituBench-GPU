source profileFunctions.sh
module load python2
curDir="${HOME}/cuda-sample-private/common/scripts"
FinalOutputDirSPMV_WideRange="${HOME}/summaryResults/sparicityRateResult/SPMV_WideRange"
FinalOutputDirSPMM_WideRange="${HOME}/summaryResults/sparicityRateResult/SPMM_WideRange"
FinalOutputDirSPMV_NarrowRange="${HOME}/summaryResults/sparicityRateResult/SPMV_NarrowRange"
FinalOutputDirSPMM_NarrowRange="${HOME}/summaryResults/sparicityRateResult/SPMM_NarrowRange"


if [ "$HOSTNAME" = "automata20" ]; then
    echo $HOSTNAME
else
    module load cuda-toolkit-10
    echo $HOSTNAME    
fi

###########################NarrowRange
#:<<'END'
rm -rf ${FinalOutputDirSPMV_NarrowRange}
rm -rf ${FinalOutputDirSPMM_NarrowRange}
mkdir ${FinalOutputDirSPMV_NarrowRange}
mkdir ${FinalOutputDirSPMM_NarrowRange}
for i in {1..20..4}
do
	#numRow=327680
	numRow=8192
	numCol=3276800
	#percent=$((0.01*$i))
	percent=$( python -c "from math import ceil; print 0.01*$i")
	echo $percent
	printf -v iFormated "%02d\n" $i
        CommomFlags=" --NumRow=${numRow} --NumCol=${numCol} --percentage=${percent}  "
#        set -x  #debugging is turned off inside the function
        prof_and_parse "9_myExamples/SPMV" "spmv"  "spmv${iFormated}" ${FinalOutputDirSPMV_NarrowRange} "\${CommomFlags}" "100000rows_${i}_percent" 
#        set -x  # debugging is turned off inside the function
#        prof_and_parse "9_myExamples/SPMM" "spmm"  "spmm${iFormated}" ${FinalOutputDirSPMM_NarrowRange} "\${CommomFlags}" "100000rows_${i}_percent"
    done
#END
###########################WideRange
:<<'END'
#start=((2**(-20)) 
startV=$( python -c "print 2.0**(-20)")
echo startV is $startV
rm -rf ${FinalOutputDirSPMV_WideRange}
rm -rf ${FinalOutputDirSPMM_WideRange}
mkdir ${FinalOutputDirSPMV_WideRange}
mkdir ${FinalOutputDirSPMM_WideRange}
for i in {1..20..1}
do
#	numRow=((32768*(2**10)/(2**($i/2))))
	numRow=$( python -c "from math import ceil; print ceil(32768*(2**10)/(2**($i/2)))")
	#percent=(($start*(2**$i))
	percent=$( python -c "print $startV*(2**$i)")
	echo $percent
	printf -v iFormated "%02d\n" $i
#:<<'END'
    CommomFlags=" --Num=${numRow} --percentage=${percent}  "
    #set -x
    prof_and_parse "9_myExamples/SPMV" "spmv"  "spmv${iFormated}" ${FinalOutputDirSPMV_WideRange} "\${CommomFlags}" "100000rows_${i}_percent" 
    prof_and_parse "9_myExamples/SPMM" "spmm"  "spmm${iFormated}" ${FinalOutputDirSPMM_WideRange} "\${CommomFlags}" "100000rows_${i}_percent"   
    #set +x
#END

done
END
