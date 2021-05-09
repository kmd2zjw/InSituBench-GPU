run_and_parse(){
eval CodeDir="$1"
eval method="$2"
eval operation="$3"
eval fileName="$4"
eval flag="$5"
eval flag_string="$6"

####################
cd "/zf14/ml2au/cuda-sample-private/${CodeDir}"
make clean
make
local tmpFinalOutputFile="tmpParsedResult.csv"
rm -f ${tmpFinalOutputFile}
echo "./${method}.out"  ${flag} 
./${method}.out  ${flag}  > ${method}${flag_string}.txt
########################
grep 'GBInputPerSecond' ${method}${flag_string}.txt | while read -r line ; do
    (echo $line | awk '{printf "%s \n",$3}') >>${tmpFinalOutputFile}
done
perfVar=$(tail -n 1 "${tmpFinalOutputFile}")
echo $perfVar
#####################
grep 'TimeOfProcessINmsec' ${method}${flag_string}.txt | while read -r line ; do
    (echo $line | awk '{printf "%s \n",$3}') >>${tmpFinalOutputFile}
done
TimeVar=$(tail -n 1 "${tmpFinalOutputFile}")
#####################
grep 'GBInputSize' ${method}${flag_string}.txt | while read -r line ; do
    (echo $line | awk '{printf "%s \n",$3}') >>${tmpFinalOutputFile}
done
SizeVar=$(tail -n 1 "${tmpFinalOutputFile}")
############################
if [ ! -f ${fileName} ]; then
    (echo -ne "HOSTNAME,";echo -ne "method,"; echo -ne "operation,";  echo -ne "perfVar,";echo -ne "TimeVar,";echo "SizeVar") >> ${fileName}
fi
(echo -ne "${HOSTNAME},";echo -ne "${method},"; echo -ne "${operation},";  echo -ne "${perfVar},";echo -ne "${TimeVar},";echo "${SizeVar}") >> ${fileName}
}


