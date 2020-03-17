## Matlab implementation of several channel coding 

Version: Matlab 2018b with Communications Toolbox 7.0 was used.

#### LDPC (Low Density Parity Check Code)

* [LDPC128.m](https://github.com/liying8040/channel_coding/blob/master/low_density_parity_check_code/LDPC128.m) : 128 QAM, 1/2 code rate
* [LDPC256.m](https://github.com/liying8040/channel_coding/blob/master/low_density_parity_check_code/LDPC256.m) : 256 QAM, 1/2 code rate

* [LDPC1024.m](https://github.com/liying8040/channel_coding/blob/master/low_density_parity_check_code/LDPC1024.m) : 1024 QAM, 1/2 code rate

The length of codeword is fixed at 64800 bits based on DVB-S.2 Standard. By changing different code rates in the corresponding Matlab file, we can get:

<img src=".\img\ldpc.png" style="zoom:50%;" />



#### Neural_Network_with_BP

* This project is a implementation of the paper "Learning to Decode Linear Codes Using Deep Learning".

#### Convolutional code

* convolutional_1_2.m : 1/2 code rate, 128QAM

* convolutional_2_3.m：2/3 code rate, 128QAM

* convolutional_3_4.m：3/4 code rate, 128QAM

By changing different modulations in the corresponding Matlab file, we can get:

<img src=".\img\conv.png" style="zoom:50%;" />