#!/usr/bin/env python
# coding: utf-8

# In[4]:

# TODO
#
# check test accuracy 
# SS2SR:
# check that when you sparsify you don't redure to one value, otherwise s goes to zero and the sampled values go to infinty by dividing by zero
#
# sparsification decided at the PS and the step t-1
# skip small layers for speeding up things
#  
# check bach role#
# step 2
# properly save models for when we change paraters 
#
# step 3
# adapt rate per layer?
# 



#from matplotlib import pyplot as plt
import random as rnd
import numpy as np
import collections
import os
import math

from utils_quantize import *
#from kmeans import *


# tf and keras
import tensorflow as tf
#import pyclustering
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
#from sklearn.cluster import KMeans
#from pyclustering.cluster.kmeans import kmeans
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.integrate import quad

#------------------------------
# DNN settings
learning_rate = 0.0001
epochs = 3

batch = 32
iterations = 10

sparsification_percentage = 45.2


# sparse_gradient[0].shape

layers_to_be_compressed=np.array([6,12,18,24,30,36,42])

# compression_type="uniform scalar"
compression_type="k-means"
#------------------------------
# channel settings

number_of_users = 2

# un-used
# TODO
# add function to compute the rates attainable over the symmetric MAC computation thing
#
#power=150
#noise_variance=0.5
#
#channel_coefficients=np.array([10,5])


# In[5]:

model = tf.keras.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
  layers.BatchNormalization(),
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.3),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
  layers.BatchNormalization(),
  layers.Dropout(0.5),
  layers.Dense(10, activation='softmax'),
])
opt = Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# In[7]:

classes = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck",
}


# In[8]:

# SR2NSS
# too many epocs in between exchanges: the accuracy already becomes 70% locally
#
# let's try less and see if
# epochs = 20


# In[9]:


def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation


# In[10]:


def top_k_sparsificate_model_weights_tf(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
    #TODO
    # same as weight.reshape.size[0] ? better make it more general
    # write as in 183
    k_th_element = tmp_list[int(fraction*552874)-1] # 552874 is the number of parameters of the CNNs!
    new_weights = []
    for el in weights:
        original_shape = el.shape
        reshaped_el = el.reshape((-1))
        for i in range(len(reshaped_el)):
            if abs(reshaped_el[i]) < k_th_element:
                reshaped_el[i] = 0.0
        new_weights.append(reshaped_el.reshape(original_shape))
    return new_weights

def pdf_gennorm(x, a, m, b):
  return stats.gennorm.pdf(x,a,m,b)

def update_centers_magnitude_distance(data, R, iterations_kmeans):
    a, m, b = stats.gennorm.fit(data)
    xmin, xmax = min(data), max(data)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(centers_init)
    thresholds_update = np.copy(thresholds_init)
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** 3 * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** 2 * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        centers_update[0] = np.divide(integ_nom, integ_denom)
        for j in range(len(centers_init) - 2):
            integ_nom_update = \
            quad(lambda x: x ** 3 * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** 2 * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            centers_update[j + 1] = np.divide(integ_nom_update, integ_denom_update)
        integ_nom_final = \
        quad(lambda x: x ** 3 * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** 2 * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
        return thresholds_update, centers_update




# In[11]:


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
num_classes = len(classes)

# normalize to one
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# categorical loss enropy
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# FL setting 
size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users,size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users,size_of_user_ds,10))
for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    

# -----------------------------------------------------------------------------
# set up the file saving
# -----------------------------------------------------------------------------


#
## prepare PD
#import pandas as pd
#
#from datetime import datetime
#
#
#now = datetime.now() # current date and time
#
#date_time = now.strftime("%Y_%m_%d_%H_%M")
#             
#file_name='sim/sim_'+date_time+'.xlsx'
#
#dict_data = ["SNR", "Pue_1 AMP", "Pue_2 AMP","Pue_1 ADMM", "Pue_2 ADMM"]
#data_pd = pd.DataFrame(columns=dict_data )
#
#simCount=100
#
#pd_settings=  pd.DataFrame([[simCount,K1, K2, L,J,M,n,2*K1 ,2*K2]],columns=['simCount','K1', 'K2', 'L','J','M','n','listSize1' ,'listSize2'])
#
#
#from openpyxl import load_workbook
#
#with pd.ExcelWriter(file_name) as writer:
#        
#        # add simulation datasTrue1[i*M:(i+1)*M
#       pd_settings.to_excel(writer, sheet_name = '0', index = False, header = False)
#
#
## ------------------
#       When saving
#       
#          data_tmp = pd.DataFrame([[EbNodB, errorRate1_AMP[ss], errorRate2_AMP[ss], errorRate1_ADMM[ss], errorRate2_ADMM[ss] ]],columns=dict_data)    
#    data_pd = data_pd.append(data_tmp)
#       
#
#
#book = load_workbook(file_name)
#writer = pd.ExcelWriter(file_name, engine='openpyxl')
#writer.book = book # <---------------------------- piece i do not understand
#data_pd.to_excel(writer, sheet_name="amp vs admm", index=None)
#writer.save()

# -----------------------------------------------------------------------------
# set up the TCQ quantizer 
# -----------------------------------------------------------------------------
    
# # indexed as [rate][memory]
# lc_coeff =  getLCCoeff()

# # SR2SS 
# # you can allocate differently for each dimension of an array.
# # in this way you don't have to loop over all layers of the network


d=1
# # indexed by dimenions of the input, so far we go through each layer separately
rate = np.array([d])
rate[0] = 4

memory = np.array([d])
memory[0] = 4


# # this is an array too, 
# # indexed as [rate][memory]
# c_scale = np.ones([10,10])

# -----------------------------------------------------------------------------
iter = 0
with open("histogram-kmeans-R4.txt", "w") as outfile:
    for _ in range(iterations):
      print('')
      print("************ Iteration " + str(_) + " ************")
      _, accuracy = model.evaluate(X_test, Y_test)
      print('Test accuracy BEFORE training is', accuracy)
      #model.save("multi-user model-before training-TCQ_iter"+str(iter)+".h5")
      iter = iter + 1
      wc = model.get_weights()
      sum_terms = []

      for i in range(number_of_users):
        X_train_u = train_data_X[i]
        Y_train_u = train_data_Y[i]
        shuffler = np.random.permutation(len(X_train_u))
        X_train_u = X_train_u[shuffler]
        Y_train_u = Y_train_u[shuffler]
        X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)

        print('user->',i)
        print(len(X_train_u))
        history = model.fit(x=X_train_u,y=Y_train_u,
                              epochs = epochs,
                              batch_size = batch,
                              validation_data=(X_validation_u, Y_validation_u)
                              )
        #model.save("multi-user model-after training-TCQ_iter" + str(iter)+"_user"+str(i) + ".h5")

        _, accuracy = model.evaluate(X_test, Y_test)


        # TODO
        # i'd use a more meaningful name ^_^
        wu = model.get_weights()


        nu = len(Y_train_u)+len(Y_validation_u)
        frac = nu/len(Y_train)

        # approx gradient with model difference
        gradient = [np.subtract(wu[i], wc[i]) for i in range(len(wu))]
        sparse_gradient = top_k_sparsificate_model_weights_tf(gradient, sparsification_percentage/100)
        #sparse_gradient = random_sparsificate_model_weights_tf(gradient, samples_fraction)

        #for j in range(len(sparse_gradient)):
        layer_index = 1
        for j in layers_to_be_compressed:
          # the size of this is 44
          # I would skip all the layers that have a small size.
          # only compress the ones in layers_to_be_compressed
          gradient_shape = np.shape(sparse_gradient[j])
          gradient_size = np.size(sparse_gradient[j])
          gradient_reshape = np.reshape(sparse_gradient[j],(gradient_size,))
          non_zero_indices = tf.where( gradient_reshape != 0).numpy()


          print("Layer",j,": entries to compress:",non_zero_indices.size )

          #SR2SS
          # i would say >1000, no need to worry about the small dimensions her
          if (non_zero_indices.size > 1):
              seq = gradient_reshape[np.transpose(non_zero_indices)[0]]

              #saving histogram before compression
              seq_max = np.amax(seq)
              seq_min = np.amin(seq)
              step_size_seq = (seq_max - seq_min) / 100
              bins_array_seq = np.arange(seq_min, seq_max, step_size_seq)
              hist_before, bin_edges_before = np.histogram(seq, bins=bins_array_seq)
              if ((j== 6) & (i==0) & (iter==10)):
                  np.savetxt(outfile,[1],header='#layer6-before comp')
                  for bin_index in range(len(bin_edges_before)-1):
                      np.savetxt(outfile, [[bin_edges_before[bin_index],hist_before[bin_index]],],fmt='%10.3e', delimiter=',')
              if ((j== 24) & (i==0) & (iter==10)):
                  np.savetxt(outfile,[2],header='#layer24-before comp')
                  for bin_index in range(len(bin_edges_before)-1):
                      np.savetxt(outfile, [[bin_edges_before[bin_index],hist_before[bin_index]],],fmt='%10.3e',delimiter =',')
              if ((j == 42) & (i == 0) & (iter == 10)):
                  np.savetxt(outfile, [3], header='#layer42-before comp')
                  for bin_index in range(len(bin_edges_before) - 1):
                      np.savetxt(outfile, [[bin_edges_before[bin_index], hist_before[bin_index]],], fmt='%10.3e',delimiter=',')
              #fig1 = plt.figure()
              #ax1 = fig1.add_subplot(1, 1, 1)
              #ax1.hist(seq, bins=bins_array_seq)
              #plt.xlabel('bins')
              #plt.ylabel('histogram of original data')
              #fig1.savefig('hist-before compression-' + 'Iter' + str(iter) + '-Layer' + str(j) + '.png')


              # SR2SS
              # the compress method already does the fitting to the gaussian
              #

    #          lc_coeff_f =   getLCCoeff()
    #          mu = np.mean(seq)
    #          s = np.var(seq)
    #          seq_scaled = np.divide(seq-mu,np.power(s,0.5))
    #          mu_scaled = np.mean(seq_scaled)
    #          s_scaled = np.var(seq_scaled)
    #
              # define memory and rate globally

    #
    #          seq_enc = quan.encode(seq_scaled)
    #          seq_dec = recon.decode(seq_enc)
    #
              #SR2SS
              # needs an array in input
              # if pipeing dimensions, change 1 to d
              #seq=seq.reshape([len(seq),1])
              #seq_dec = np.zeros(np.shape(seq))
              #if compression_type=="TCQ":
                  #indices_positive = tf.where(seq > 0).numpy()
                  #indices_negative = tf.where(seq < 0).numpy()
                  #seq_positive = seq[np.transpose(indices_positive)[0]]
                  #seq_negative = seq[np.transpose(indices_negative)[0]]

                  #seq_positive_enc, sqnr_positive, mu_positive, s_positive= compress(seq_positive, rate, memory, lc_coeff, c_scale)
                  #seq_positive_dec = decompress(seq_positive_enc, rate, memory, lc_coeff, mu_positive, s_positive, c_scale)

                  #seq_negative_enc, sqnr_negative, mu_negative, s_negative = compress(seq_negative, rate, memory, lc_coeff,  c_scale)
                  #seq_negative_dec = decompress(seq_negative_enc, rate, memory, lc_coeff, mu_negative, s_negative, c_scale)

                  #np.put(seq_dec,np.transpose(indices_positive)[0],seq_positive_dec)
                  #np.put(seq_dec,np.transpose(indices_negative)[0],seq_negative_dec)
                  #seq = seq.reshape([len(seq), 1])
                  #seq_enc, sqnr, mu, s = compress(seq, rate, memory, lc_coeff, c_scale)
                  #seq_dec = decompress(seq_enc, rate, memory, lc_coeff, mu, s, c_scale)


              elif  compression_type=="uniform scalar":

                  seq_enc, uni_max, uni_min= compress_uni_scalar(seq, rate)


                  seq_dec = decompress_uni_scalar(seq_enc, rate, uni_max, uni_min)

              elif  compression_type=="gaussian scalar":
                 seq_enc, mu, s = gaussian_compress(seq, rate[0])
                 seq_dec = decompress_gaussian(seq_enc, mu, s)

              # SR2SS
              # explore large d, use K-metoids from here using custum metric
              # https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
              # since it also has lower complexity?







              elif compression_type=="k-means":
                      thresholds, quantization_centers = update_centers_magnitude_distance(data=seq, R=rate[0], iterations_kmeans=100000)
                      thresholds_sorted = np.sort(thresholds)
                      labels = np.digitize(seq,thresholds_sorted)
                      index_labels_false = np.where(labels == 2**rate[0])
                      labels[index_labels_false] = 2**rate[0]-1
                      seq_dec = quantization_centers[labels]
                  # SR2SS
                  # need to stare overything: see how it changes over layer and over time

              elif compression_type == "optimal compression":
                  seq_enc , mu, s = optimal_compress(seq,rate)
                  seq_dec = decompress_gaussian(seq_enc, mu, s)


              elif compression_type == "no compression":
                  seq_dec = seq

              elif compression_type == "no compression with float16 conversion":
                  seq_dec = seq.astype(np.float16)


              # compress_decompress(type='TCQ')

              #plot the histogram of data



              #saving the histogram after compression
              dec_max = np.amax(seq_dec)
              dec_min = np.amin(seq_dec)
              step_size = (dec_max - dec_min) / 100
              bins_array_dec = np.arange(dec_min, dec_max, step_size)
              hist_after, bin_edges_after = np.histogram(seq_dec, bins=bins_array_dec)

              if ((j== 6) & (i==0) & (iter==10)):
                  np.savetxt(outfile,[1],header='#layer6-after comp-histogram')
                  for bin_index in range(len(bin_edges_after)-1):
                      np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e', delimiter=',')
              if ((j== 24) & (i==0) & (iter==10)):
                  np.savetxt(outfile,[2],header='#layer24-after comp-histogram')
                  for bin_index in range(len(bin_edges_after)-1):
                      np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e',delimiter =',')
              if ((j == 42) & (i == 0) & (iter == 10)):
                  np.savetxt(outfile, [3], header='#layer42-after comp-histogram')
                  for bin_index in range(len(bin_edges_after) - 1):
                      np.savetxt(outfile, [[bin_edges_after[bin_index], hist_after[bin_index]],], fmt='%10.3e',delimiter=',')

              #unique_labels, unique_indices, counts = np.unique(seq_dec,return_index=True,return_counts=True)
              #if ((j== 12) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[1],header='#layer12-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e',delimiter=',')
              #if ((j== 24) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[2],header='#layer24-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e', delimiter =',')
              #if ((j == 42) & (i == 0) & (iter == 10)):
              #    np.savetxt(outfile, [3], header='#layer42-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],], fmt='%10.3e',delimiter=',')
              #np.savetxt(outfile, [bin_edges_after])
              #np.savetxt(outfile, [hist_after])
              #fig = plt.figure()
              #ax = fig.add_subplot(1, 1, 1)
              #ax.hist(seq_dec, bins=bins_array)
              #plt.xlabel('bins')
              #plt.ylabel('histogram of quantized data')
              #fig.savefig('hist-after compression-'+'Iter'+str(iter)+'-Layer'+ str(j)+'.png')


              # SR2SS check the noise distribution
              #while False:
              #np.savetxt('original_gradient_seq_iteration'+str(iter)+'_user'+str(i)+'_layer'+str(j)+'.txt', seq)
              #np.savetxt('compressed_gradient_seq_iteration'+str(iter)+'_user'+str(i)+'_layer'+str(j)+'.txt', seq_dec)
              #plt.hist(seq,int(np.sqrt(len(seq/10))),label='data')
              #plt.hist(seq,int(np.sqrt(len(seq))),label='data')
              #plt.hist(err,int(np.sqrt(len(seq))),label='err')
              #plt.legend()

              # assume noiseless, rate limited transmission,
              # assume we send mu, s noiselessly
              #
              # quan = SR_LC_Int_Quantizer(memory=5, rate=4, lc_coeff=lc_coeff_f, mu=mu_scaled, s=s_scaled, c_scale = 1, distortion_measure = 'mse')
              # recon = SR_LC_Int_Reconstructor(memory=5, rate=4, lc_coeff=lc_coeff_f, mu=mu_scaled, s=s_scaled, c_scale = 1)
              #

              # no need now
              # gradient_reshape_quantized = np.multiply(seq_dec,np.power(s,0.5))+mu
              # np.put(gradient_reshape, np.transpose(non_zero_indices)[0], gradient_reshape_quantized)

              np.put(gradient_reshape, np.transpose(non_zero_indices)[0], seq_dec)

              sparse_gradient[j] = gradient_reshape.reshape(gradient_shape)
              layer_index = layer_index+1


        # this is the PS part

        sum_terms.append(np.multiply(frac,sparse_gradient))
        model.set_weights(wc)



      update = sum_terms[0]
      for i in range(1, len(sum_terms)): # could do better...
        tmp = sum_terms[i]
        update = [np.add(tmp[j], update[j]) for j in range(len(update))]
      new_weights = [np.add(wc[i], update[i]) for i in range(len(wc))]
      model.set_weights(new_weights)

      # check test accuracy
      results = model.evaluate(X_test, Y_test)
      # check the performance at the PS, monitor the noise
      print('Test accuracy AFTER PS aggregation',results[1])
      #np.savetxt(outfile, [int(iter),results[1]])




# In[ ]:




