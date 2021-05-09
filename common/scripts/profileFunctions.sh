
###############################################
prof_and_parse(){
set +x
eval CodeDir="$1"
eval method="$2"
eval operation="$3"
eval outputDirectoryName="$4"
eval flag="$5"
eval flag_string="$6"
####################
echo "./${method}.out"  ${flag} 
####################
profCommand=nvprof
input="/zf14/ml2au/cuda-sample-private/common/scripts/ListOfMetric.list"
directoryForMetrics=${outputDirectoryName}/${operation}
rm -rf ${directoryForMetrics}
mkdir -p ${directoryForMetrics}
nIteration=30
while IFS= read -r metricName
do
	[ -z "$metricName" ] && continue
	profFlags="--csv --log-file ${directoryForMetrics}/${metricName}.csv  --metrics ${metricName}"
	echo $profFlags
	####################
	cd "/zf14/ml2au/cuda-sample-private/${CodeDir}"
	#make clean
	#make
	if [[ "${metricName}" == "powerConsumption" ]]; then
		${profCommand} --csv --system-profiling on  --log-file ${directoryForMetrics}/powerConsumption.csv  ${method}.out  ${flag} --nIter=${nIteration} --NoWarmUp
	elif [[ "${metricName}" == "executionTime" ]]; then	
		${profCommand} --csv --log-file ${directoryForMetrics}/executionTime.csv ${method}.out  ${flag}  --nIter=${nIteration} --NoWarmUp
		./${method}.out  ${flag}  --nIter=${nIteration} --NoWarmUp > ${directoryForMetrics}/executionTime.tmp
		grep 'nIter =' ${directoryForMetrics}/executionTime.tmp | while read -r line ; do
   		 (echo $line | awk '{printf "%s \n",$3}') >${directoryForMetrics}/executionTimeRunResult.tmp
		done
		grep 'TimeOfProcessINmsec' ${directoryForMetrics}/executionTime.tmp | while read -r line ; do
   		 (echo $line | awk '{printf "%s \n",$3}') >>${directoryForMetrics}/executionTimeRunResult.tmp
		done
	else 
		${profCommand} ${profFlags} ${method}.out  ${flag} --nIter=1 --NoWarmUp
	fi
	
	########################
done < "$input"
set -x
}
# nvprof --csv --system-profiling on  --log-file human-readable-output_A.log  ./matrixMulCUBLAS.out --sizemult=200 --IsVector=1 --nIter=1 --NoWarmUp

