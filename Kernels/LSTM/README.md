## Optimized LSTM forward pass in CUDA

This code implements an LSTM forward pass in CUDA using cuBLAS. It uses cuRand to initialize the inputs and parameters.

The PERFOPTS define allows the user to select which optimizations are enabled.
more explanation in "https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/"
and



Optimizing Performance of Recurrent Neural Networks on GPUs(https://arxiv.org/abs/1604.01946)
It is a bitmask taking values from 0 to 31 with each bit defining whether a particular optimization is enabled.
