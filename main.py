


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
from scipy.integrate import quad

#------------------------------
# DNN settings
learning_rate = 0.0001
epochs = 3

batch = 32
iterations = 10

sparsification_percentage = 45.2


# DNN layers to be compressed

layers_to_be_compressed=np.array([6,12,18,24,30,36,42])

# compression_type="uniform scalar"
compression_type="k-means"
#------------------------------
# channel settings

number_of_users = 2



# definition of the DNN model

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

# definition of the classes

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





# splitting data into train and test datasets


def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation


# topK sparsification function


def top_k_sparsificate_model_weights_tf(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
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

#GenNorm pdf function  
  
def pdf_gennorm(x, a, m, b):
  return stats.gennorm.pdf(x,a,m,b)

#Kmeans algorithm adapted to the distortion metrix \sum |g|^M|g-\hat{g}|^2

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
    



d=1
# # compression rate
rate = np.array([d])
rate[0] = 4






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

          
          
          if (non_zero_indices.size > 1):
              seq = gradient_reshape[np.transpose(non_zero_indices)[0]]

              #saving histogram of data before compression
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
        


              if  compression_type=="uniform scalar":

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
                  

              elif compression_type == "optimal compression":
                  seq_enc , mu, s = optimal_compress(seq,rate)
                  seq_dec = decompress_gaussian(seq_enc, mu, s)


              elif compression_type == "no compression":
                  seq_dec = seq

              elif compression_type == "no compression with float16 conversion":
                  seq_dec = seq.astype(np.float16)





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









