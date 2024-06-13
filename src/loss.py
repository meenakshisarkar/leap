#Code developed by Meenakshi Sarkar, IISc for the project pgvg
import sys
import math
import numpy as np 
import tensorflow as tf
tf.random.set_seed(77)
from tensorflow import keras
from keras.losses import Loss
# tf.compat.v1.enable_eager_execution()
# from joblib import Parallel, delayed

from tensorflow.python.framework import ops

from utils import *

def reduce_mean(inputs,global_batch_size):
    """ return inputs mean with respect to the global_batch_size """
    return tf.reduce_sum(inputs) / global_batch_size

# def KLD_loss(mu_x,v_x,mu_y,v_y,global_batch_size):
#     """ Kl_divergence loss """
#     first= tf.math.log(tf.linalg.det(tf.linalg.diag(v_y))/tf.linalg.det(tf.linalg.diag(v_x)))
#     second= tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(tf.linalg.diag(v_y)),tf.linalg.diag(v_x)))
#     third = tf.squeeze(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(tf.expand_dims(mu_x-mu_y,1),perm=[0,2,1]),tf.linalg.inv(tf.linalg.diag(v_y))), tf.expand_dims(mu_x-mu_y,1) ))
#     return tf.reduce_sum( 1/2*(first + second + third - mu_x.shape[-1])) / global_batch_size



class KL_Divergence(Loss):
  def __init__(self,batch_size=8,name="KL_Divergence",**kwargs):
      super().__init__(name=name, **kwargs)
      self.batch_size=batch_size


  def __call__(self, mu_pred=None, logvar_pred=None, mu_true=None, logvar_true=None):
    # shape=mu_pred.shape
    first_term= (logvar_true-logvar_pred)/2
    # second_term=  (tf.math.exp(logvar_pred) + (mu_true-mu_pred)**2 ) / (2*tf.math.exp(logvar_true)) 
    second_term=  (tf.math.exp(logvar_pred) + (mu_pred-mu_true)**2 ) / (2*tf.math.exp(logvar_true)) 
    kld_loss= tf.reduce_mean(first_term + second_term -1/2,axis=list(range(2, len( mu_pred.shape))))
    # print(kld_loss)
    # kld_loss = tf.reduce_sum(kld_loss,axis=-1)
    kld_loss = tf.reduce_sum(kld_loss,axis=list(range(1, len(kld_loss.shape)) ))
    # print(kld_loss)
    # kld_loss=kld_loss*shape[1]
    return kld_loss
    # print(kld_loss)
    # return reduce_mean(kld_loss,self.batch_size)
# 



class sigmoid_cross_entropy_with_logits(Loss):
  def __init__(self,name="cross_entropy",batch_size=8,**kwargs):
      super().__init__(name=name, **kwargs)
      self.batch_size=batch_size


  def call(self, logits=None, labels=None):
     loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels), axis=list(range(1, len(logits.shape))))
     return reduce_mean(loss,self.batch_size)
  
  
  


  
class recon_loss_l2(Loss):
  def __init__(self,batch_size=8,name="recon_loss_l2",**kwargs):
      super().__init__(name=name, **kwargs)
      self.batch_size=batch_size

  def call(self,gt,output):
     shape=gt.shape
     loss = tf.reduce_mean(tf.square(gt - output),axis=list(range(2, len(gt.shape))))
    #  print(loss)
    # #  loss=loss*shape[1]
     loss = tf.reduce_sum(loss,axis=list(range(1, len(loss.shape))) )
    #  print(loss)
    #  print("Loss")
    #  return reduce_mean(loss,self.batch_size)
     return loss

class recon_loss_l1(Loss):
  def __init__(self,batch_size=8,name="recon_loss_l1",**kwargs):
      super().__init__(name=name, **kwargs)
      self.batch_size=batch_size
  def call(self,gt,output):
    shape=gt.shape
    loss = tf.reduce_mean(
                    tf.abs(gt - output),axis=list(range(2, len(gt.shape))))
    # loss=loss*shape[1]
    loss = tf.reduce_sum(loss,axis=list(range(1, len(loss.shape))) )
    # print(loss)
    # return reduce_mean(loss,self.batch_size)
    return loss
  
class avg_loss(Loss):
   def __init__(self,batch_size=None,name="avg_loss",**kwargs):
      super().__init__(name=name, **kwargs)
      self.batch_size=batch_size
   def __call__(self,input):
      loss = tf.nn.compute_average_loss(input)
      # print(loss)
      return loss
  


