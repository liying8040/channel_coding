### Learning to Decode Linear Codes Using Deep Learning

**写在前面：**

​		本项目实现了论文《Learning to Decode Linear Codes Using Deep Learning》提出的基于深度神经网络的HDPC置信传播（Belief Propagation，BP）译码算法。

**国内外现状研究：**

​		对于基于稀疏因子图的线性纠错码，如低密度奇偶校验（Low-Density Parity-Check，LDPC）码 [1]，低复杂度的置信度传播BP译码算法可达到接近香农极限的性能，在移动通信领域被广泛应用。但对于中短长度的基于稠密因子图的高密度校验（High-Density Parity-Check，HDPC）码[2] [3] [4] [5]而言，由于受因子图中短环的限制，BP译码算法相较于最大似然（Maximum-Likelihood，ML）译码算法性能存在一定差距[6]，而受限于ML译码算法的高复杂度，我们在实际应用中很难获得接近ML的性能。

​		基于神经网络, 文献[7]首次提出了一种提高HDPC码的 BP译码性能的前馈神经网络译码算法，利用随机梯度下降（Stochastic Gradient Descent，SGD）算法将因子图连接边的权重训练为实数，以此替代传统的非0即1的权重，但该方案的缺陷在于运算复杂度大，且不易于硬件实现。为弥补该缺陷，文献[8]进一步提出了基于偏移最小和的神经网络译码算法，校验节点学习每条边的加性偏移参数，而不是乘法权重，在保证了与[7]获得同样性能的同时，降低了运算复杂度。文献[9]提出使用递归神经网络优化BP译码，可在高SNR区域获得了1.5dB的增益，并能够进一步降低了译码复杂度。文献[10(2)]通过将神经BP译码算法与自同构群（Automorphism Group）置换相结合，进一步提高了BP译码算法的性能。

**参考文献：**

[1] R. G. Gallager, Low Density Parity Check Codes. Cambridge, Massachusetts: M.I.T. Press, 1963.  

[2] J. Jiang and K. R. Narayanan, “Iterative soft-input soft-output decoding of reed-solomon codes by adapting the parity-check matrix,” IEEE Transactions on Information Theory, vol. 52, no. 8, pp. 3746–3756,2006.  

[3] I. Dimnik and Y. Be’ery, “Improved random redundant iterative hdpc decoding,” IEEE Transactions on Communications, vol. 57, no. 7, pp.1982–1985, 2009.  

[4] A. Yufit, A. Lifshitz, and Y. Be’ery, “Efficient linear programming decoding of hdpc codes,” IEEE Transactions on Communications,vol. 59, no. 3, pp. 758–766, 2011.  

[5] X. Zhang and P. H. Siegel, “Adaptive cut generation algorithm for improved linear programming decoding of binary linear codes,” IEEE Transactions on Information Theory, vol. 58, no. 10, pp. 6581–6594,2012.  

[6] M. Helmling, E. Rosnes, S. Ruzika, and S. Scholl, “Efficient maximum-likelihood decoding of linear block codes on binary memoryless channels,” in 2014 IEEE International Symposium on Information Theory, pp. 2589–2593, 2014.  

[7] E. Nachmani, Y. Be’ery, and D. Burshtein, “Learning to decode linear codes using deep learning,” 54th Annual Allerton Conf. on Communication, Control and Computing, 2016.  

[8] L. Lugosch and W. J. Gross, “Neural offset min-sum decoding,” in 2017 IEEE International Symposium on Information Theory, arXiv preprint arXiv:1701.05931, June 2017.  

[9] E. Nachmani, E. Marciano, L. Lugosch, W. J. Gross, D. Burshtein, and Y. Be’ery, “Deep learning methods for improved decoding of linear codes,” IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 119–131, 2018.  

[10] E. Nachmani, Y. Bachar, E. Marciano, D. Burshtein, and Y. Be’ery, “Near maximum likelihood decoding with deep learning,” arXiv preprint arXiv:1801.02726, 2018.  

------

**代码运行：**

（1）[在win10下安装Anaconda](https://blog.csdn.net/u010858605/article/details/64128466)

（2） 打开"Anaconda Prompt"，键入`activate tensorflow` 启动tensorflow环境

（3）键入`spyder` 启动spyder IDE，即可运行代码。

------

**代码说明：**

* train.py			//训练过程

  本项目以BCH(63,36)码为例，运行完train.py后，将训练好的模型保存在了save文件夹中。其中，文件63_36_10001表示BCH(63,36)在minibatch_num = 10001时的训练模型，同理，文件63_36_20001表示BCH(63,36)在minibatch_num = 20001时的训练模型。

* validation.py	//验证过程

  在validation.py中修改SNR(snr_vali)，即可得到对应的BER。

  如果在运行validation.py时报OOM ResourceExhaustedError的错误，可能是因为GPU的显存太小，或者剩余的显存太少了，将validation_num的值减少就可以了。经测试发现，当validation_num设置为10000时，在服务器上跑没有问题，在笔记本运行就会报OOM ResourceExhaustedError错，调整到了1000后，再运行就没有问题了。

**实验结果：**

| SNR (dB) |     BER     |
| :------: | :---------: |
|    1     | 0.113209524 |
|    2     | 0.081239683 |
|    3     | 0.045947619 |
|    4     | 0.019568254 |
|    5     | 0.00608254  |
|    6     | 0.001463492 |