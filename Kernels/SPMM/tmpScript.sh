#./spmm.out --NumRowA=8192  --NumColA=100000 --NumColB=8192 --percentage=0.2 
nvprof --metrics dram_read_bytes ./spmm.out --NumRowA=8192  --NumColA=10000 --NumColB=8192 --percentage=0.2 
