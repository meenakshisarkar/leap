""" this is file where all the configuration for training the network including the image sizes, batch size, past timestep and future
 or prediction time steps are provided. We also provide the dat path, model save paths here only. The main file and model file creats 
 an instance of the config object """
 #Code developed by Meenakshi Sarkar, IISc for the project pgvg

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
tf.random.set_seed(77)
# import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

import os
import numpy as np
from os.path import exists
from os import makedirs


def define_network_flags():
  flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
  flags.DEFINE_integer('batch_size',os.environ.get('BATCH_SZ'), 'Batch Size')
  flags.DEFINE_integer('image_h', os.environ.get('IMG_H'), 'image_h')
  flags.DEFINE_integer('image_w', os.environ.get('IMG_W'), 'image_w')
  flags.DEFINE_integer('n_samples', os.environ.get('N_SAMPLES'), 'n_samples')
  flags.DEFINE_float('lr', 0.0001, 'lr')
  flags.DEFINE_float('lr_pgvg', 0.000001, 'lr_pgvg')
  flags.DEFINE_float('alpha', 1.0, 'alpha')
  # flags.DEFINE_float('beta', 0.000001, 'beta') ##for kth
  # flags.DEFINE_float('beta', 0.001, 'beta') ##for kth
  flags.DEFINE_float('beta', os.environ.get('BETA'), 'beta the co-efficient of kld loss')
  flags.DEFINE_float('A_beta', os.environ.get('A_BETA'), 'beta the co-efficient of action loss')
  flags.DEFINE_float('gamma', os.environ.get('GAMMA'), 'gamma the co-efficient of kld action loss')
  flags.DEFINE_integer('c_dim', os.environ.get('C_DIM'), 'c_dim')
  flags.DEFINE_integer('h_dim', os.environ.get('H_DIM'), 'H_dim is the output and input size of the encoder and decoder NN')
  flags.DEFINE_integer('alpha_dim', os.environ.get('alpha_dim'), 'alpha_dim is the latent dimension of the action')
  flags.DEFINE_integer('h_dimA', os.environ.get('h_dimA'), 'h_dimA is the output and input size of the action encoder and decoder NN')
  flags.DEFINE_integer('hidden_dim_prior', os.environ.get('HIDDEN_DIM_PRIOR'), 'hidden_dim of the LSTM cell for prior')
  flags.DEFINE_integer('hidden_dim_pred', os.environ.get('HIDDEN_DIM_PRED'), 'hidden_dim of the LSTM cell for pred')
  flags.DEFINE_integer('hidden_dim_priorA', os.environ.get('hidden_dim_priorA'), 'no of LSTM cell for prior of action lstm')
  flags.DEFINE_integer('a_rnn', os.environ.get('A_RNN'), 'No of LSTM cell for action prediction RNN')
  flags.DEFINE_integer('a_dim', os.environ.get('A_DIM'), 'a_dim')
  flags.DEFINE_integer('past_TS', os.environ.get('PAST_TS'), 'past_TS')
  flags.DEFINE_integer('future_TS', os.environ.get('FUTURE_TS'), 'future_TS')
  flags.DEFINE_integer('test_future_TS', os.environ.get('TEST_FUTURE_TS'), 'test_future_TS')
  flags.DEFINE_integer('epochs', os.environ.get('EPOCHS'), 'Number of epochs')
  flags.DEFINE_integer('epochs_critic', os.environ.get('EPOCHS_CRITIC'), 'Number of epochs for critic')
  flags.DEFINE_integer('epochs_pgvg', os.environ.get('EPOCHS_PGVG'), 'Number of epochs for pgvg')
  flags.DEFINE_float('beta1', 0.9, 'beta1')
  # flags.DEFINE_integer('gpu', os.environ.get('GPU_ID'), 'GPU')
  flags.DEFINE_string('ckpt_G', os.environ.get('CKPT_G'), 'CKPT_G')
  flags.DEFINE_string('ckpt_P', os.environ.get('CKPT_P'), 'CKPT_P')
  flags.DEFINE_string('ckpt_C', os.environ.get('CKPT_C'), 'CKPT_C')
  flags.DEFINE_integer('filters', os.environ.get('FILTERS'), 'filters')
  flags.DEFINE_integer('latent_dim', os.environ.get('LATENT_DIM'), 'latent_dim of the prior and posterior nn')
  # flags.DEFINE_integer('a_filters', os.environ.get('A_FILTERS'), 'a_filters')
  # flags.DEFINE_integer('df_dim', 32, 'df_dim')
  flags.DEFINE_float('margin', 0.3, 'margin')
  flags.DEFINE_boolean('tf_enable', True, 'tf_Enable?')
  flags.DEFINE_boolean('skip',os.environ.get('SKIP'), 'skip')
  flags.DEFINE_boolean('reload_ckpt', os.environ.get('RELOAD_CKPT'), 'reload_ckpt')
  flags.DEFINE_boolean('G_train',os.environ.get('G_TRAIN'), 'G_train')
  flags.DEFINE_boolean('C_train', os.environ.get('C_TRAIN'), 'C_train')
  flags.DEFINE_boolean('test_G', os.environ.get('TEST_G'), 'test_G')
  flags.DEFINE_string('datapath', os.environ.get('DATAPATH'), 'Directory to  the dataset')
  # flags.DEFINE_string('model_name', 'Generator', 'Deciding model name')
  flags.DEFINE_string('model_name', os.environ.get('model_name'), 'Deciding model name')

  flags.DEFINE_string('dataset', os.environ.get('DATASET'), 'Which dataset to train on')
  
  # flags.DEFINE_string('train_mode', 'custom_loop',
  #                     'Use either "keras_fit" or "custom_loop"')

def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """

  kwargs = {
      'epochs': FLAGS.epochs,
      'epochs_critic': FLAGS.epochs_critic,
      'epochs_pgvg': FLAGS.epochs_pgvg,
      'tf_enable': FLAGS.tf_enable,
      'buffer_size': FLAGS.buffer_size,
      'batch_size': FLAGS.batch_size,
      'image_h': FLAGS.image_h,
      'image_w': FLAGS.image_w,
      'n_samples': FLAGS.n_samples,
      'model_name': FLAGS.model_name,
      'lr': FLAGS.lr,
      'lr_pgvg': FLAGS.lr_pgvg,
      'alpha': FLAGS.alpha,
      'beta': FLAGS.beta,
      'A_beta': FLAGS.A_beta,
      'gamma': FLAGS.gamma,
      'beta1': FLAGS.beta1,
      'c_dim': FLAGS.c_dim,
      'h_dim': FLAGS.h_dim,
      'h_dimA': FLAGS.h_dimA,
      'alpha_dim': FLAGS.alpha_dim,
      'hidden_dim_prior': FLAGS.hidden_dim_prior,
      'hidden_dim_pred': FLAGS.hidden_dim_pred,
      'hidden_dim_priorA': FLAGS.hidden_dim_priorA,
      'a_rnn': FLAGS.a_rnn,
      'a_dim': FLAGS.a_dim,
      'past_TS': FLAGS.past_TS,
      'future_TS': FLAGS.future_TS,
      'test_future_TS': FLAGS.test_future_TS,
      # 'gpu': FLAGS.gpu,
      'filters': FLAGS.filters,
      'latent_dim': FLAGS.latent_dim,
      # 'a_filters': FLAGS.a_filters,
      # 'df_dim': FLAGS.df_dim,
      'margin': FLAGS.margin,
      'G_train': FLAGS.G_train,
      'reload_ckpt': FLAGS.reload_ckpt,
      'skip': FLAGS.skip,
      'C_train': FLAGS.C_train,
      'test_G': FLAGS.test_G,
      'ckpt_G': FLAGS.ckpt_G,
      'ckpt_P': FLAGS.ckpt_P,
      'ckpt_C': FLAGS.ckpt_C,
      'datapath': FLAGS.datapath,
      'dataset': FLAGS.dataset
  }
  return kwargs



        

