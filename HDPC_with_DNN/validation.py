# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:42:14 2017

@author: liying
"""

import tensorflow as tf
import numpy as np

## 重置图,将Variable变量的名称置0，为了与Save时保持一致
from tensorflow.python.framework import ops
ops.reset_default_graph()
#sess = tf.InteractiveSession()

# (0).Some Parameters
#minibatch_num = 301#20000
#minibatch_size = 120
validation_num = 1000 #10000
snr_vali = 6
clip_tanh = 10.0
#learning_rate = 0.001
num_cpu_core = 12
#model_path = "./save/63_36_20001/model_63.ckpt"  
model_path = "./save/63_36_10001/model_63.ckpt"  

# H : Parity Check Matrix 

#H = np.loadtxt('BCH15_11.txt')
#G = np.loadtxt('G15_11.txt')

H = np.loadtxt('BCH63_36.txt')
G = np.loadtxt('G63_36.txt')

#H = np.loadtxt('BCH63_45.txt')
#G = np.loadtxt('G63_45.txt')
#
#H = np.loadtxt('BCH127_106.txt')
#G = np.loadtxt('G127_106.txt')

k = len(G)
n = len(G[0])
r = n - k
rate = 1.0*k/n
# pos: the first column is C; the second column is V
pos = []
for i in range(len(H)):
    for j in range(len(H[0])):
        if H[i][j] == 1:
            pos.append([i,j])

# the number nodes in input layer and output layer
v_size = len(H[0]) # 15 

# the number nodes in each hidden layers
e_size = len(pos)    # 32

# array1：from input layer to odd hidden layer; (has W)
input_to_oddH = np.zeros([v_size, e_size], dtype=np.float32)
for e in range(e_size):
    input_to_oddH[pos[e][1]][e] = 1
    
# array2：from odd hidden layer to even hidden layer; C is unchanged; no W
oddH_to_evenH = np.zeros([e_size, e_size], dtype=np.float32)
for ex in range(e_size):
    for ey in range(e_size):
        if (pos[ey][0] == pos[ex][0]) and (ey!=ex):
            oddH_to_evenH[ex][ey] = 1

# array3：from even hidden layer to odd hidden layer; v is unchanged; has W
evenH_to_oddH = np.zeros([e_size, e_size], dtype=np.float32)
for ex in range(e_size):
    for ey in range(e_size):
        if (pos[ey][1] == pos[ex][1]) and (ey!=ex):
            evenH_to_oddH[ex][ey] = 1

# array4: from input layer to output layer; has W
input_to_output = np.eye(v_size, dtype=np.float32)

# array5: from the last hidden layer to the output layer
oddH_to_output = np.transpose(input_to_oddH)

def add_even_layer(inputs_e, flag_clip):
    even_layer_1 = tf.tile(inputs_e, multiples = [1, e_size])
    even_layer_2 = tf.multiply(even_layer_1, tf.reshape(oddH_to_evenH.transpose(),[-1]))
    even_layer_3 = tf.reshape(even_layer_2, [validation_num, e_size, e_size])
    even_layer_4 = tf.add(even_layer_3, 1-tf.to_float(tf.abs(even_layer_3)>0))
    even_layer_5 = tf.reduce_prod(even_layer_4,reduction_indices = 2)
    if flag_clip == 1:
        even_layer_5 = tf.clip_by_value(even_layer_5, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)              
    even_outputs = tf.log(tf.div(1 + even_layer_5, 1 - even_layer_5))
    return even_outputs

         
def add_odd_layer(inputs_v, inputs_e):
    Weights_v = tf.Variable(input_to_oddH, tf.float32)
    Weights_e = tf.Variable(evenH_to_oddH, tf.float32)
    odd_layer_1 = tf.matmul(inputs_v, tf.multiply(Weights_v, input_to_oddH)) # None*e_size
    odd_layer_2 = tf.matmul(inputs_e, tf.multiply(Weights_e, evenH_to_oddH)) # None*e_size
    odd_layer_3 = tf.add(odd_layer_1, odd_layer_2)
    odd_layer_4 = 0.5*tf.clip_by_value(odd_layer_3, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
    odd_outputs = tf.tanh(odd_layer_4)
    return odd_outputs


def add_output_layer(inputs_v, inputs_e):
    Weights_v = tf.Variable(input_to_output, tf.float32)
    Weights_e = tf.Variable(oddH_to_output, tf.float32)
    final_layer_1 = tf.matmul(inputs_v, tf.multiply(Weights_v, input_to_output))
    final_layer_2 = tf.matmul(inputs_e, tf.multiply(Weights_e, oddH_to_output))
    final_outputs = tf.add(final_layer_1, final_layer_2)
    return final_outputs


# BCH Encode
data = np.random.randint(0,2,size = [validation_num, k])
encodeData = np.mod(np.matmul(data, G),2)
modSignal = 2 * encodeData - 1
Es = 1
snr = 10**(snr_vali/10.0)
npower = Es / (2 * snr * rate)            # npower = sigma2
N = np.zeros([validation_num, n])
for i in range(validation_num):
    N[i] = np.sqrt(npower) * np.random.randn(n)

#N = np.sqrt(npower) * np.random.randn([validation_num, n])
receivedSignal = modSignal + N
demodSignal = 2 * receivedSignal / npower


# (2).Define placeholder for inputs to network 
xs = tf.placeholder(tf.float32, [None, v_size])
#ys = tf.placeholder(tf.float32, [None, v_size])
    
# (3).Add layers
flag_clip = 1
odd1 = add_odd_layer(xs, tf.zeros([tf.shape(xs)[0],e_size]))
even2 = add_even_layer(odd1, flag_clip)

flag_clip = 0
odd3 = add_odd_layer(xs,even2)
even4 = add_even_layer(odd3, flag_clip)

odd5 = add_odd_layer(xs,even4)
even6 = add_even_layer(odd5, flag_clip)

odd7 = add_odd_layer(xs,even6)
even8 = add_even_layer(odd7, flag_clip)

odd9 = add_odd_layer(xs,even8)
even10 = add_even_layer(odd9, flag_clip)
prediction = add_output_layer(xs, even10)


# (4.5) Validation part
results = tf.sigmoid(prediction*1000000)
bool_equal = tf.equal(results, encodeData)
right = tf.reduce_sum(tf.cast(bool_equal, tf.float32))

saver = tf.train.Saver()

config = tf.ConfigProto(device_count={"CPU": num_cpu_core }, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 1,   
                intra_op_parallelism_threads = 1,  
                log_device_placement=True) 

with tf.Session(config = config) as sess:
    
    saver.restore(sess,model_path)
    results = sess.run(results, feed_dict={xs:demodSignal})
    right = sess.run(right, feed_dict={xs:demodSignal})
        
    print('BER:',(validation_num * n - right)/(validation_num * n))
