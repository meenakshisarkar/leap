#Code developed by Meenakshi Sarkar, IISc for the project pgvg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl import app
import tensorflow as tf
tf.random.set_seed(77)

import config 
import data.dataloader as D
from os.path import exists
from os import makedirs
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration()])
      tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=30024)])
  except RuntimeError as e:
    print(e)

def run_main(argv):
  """Passes the flags to main.

  Args:
    argv: argv
  """
  del argv
  kwargs = config.flags_dict()
  main(**kwargs)


def main(epochs=2500,
      epochs_critic=1000,
      epochs_pgvg=1000,
      tf_enable=True,
      buffer_size=5000,
      batch_size=8,
      image_h=64,
      image_w=64,
      model_name="Generator",
      lr=0.0001,
      lr_pgvg=0.00000001,
      alpha=1,
      beta=0.00001,
      A_beta=0.00001,
      gamma=0.001,
      beta1=0.5,
      n_samples=50,
      c_dim=3,
      a_dim=2,
      alpha_dim=8,
      h_dim=128,
      h_dimA=16,
      hidden_dim_prior=256,
      hidden_dim_pred=256,
      hidden_dim_priorA=32,
      a_rnn=32,
      past_TS=5,
      future_TS=10,
      test_future_TS=20,
      # gpu=0,
      filters=32,
      # a_filters=32,
      latent_dim=128,
      # df_dim=32,
      margin=0.3,
      ckpt_G=63,
      ckpt_P=63,
      ckpt_C=25,
      G_train=True,
      skip=False,
      reload_ckpt=False,
      C_train=True,
      test_G=False,
      datapath=None,
      dataset=None):
   
  strategy = tf.distribute.MirroredStrategy() 
  # strategy=tf.distribute.OneDeviceStrategy(device="/gpu:0")
  global_batch_size = strategy.num_replicas_in_sync*batch_size

  print(f'Number of devices: {strategy.num_replicas_in_sync}')
  print(f'Learning rate for this training instance {lr} and the beta for KLD loss is {beta} and action_loss parameter: {A_beta}')
  print(f"The dimention of the latent variable z is {latent_dim} and image hidden dimentsion {h_dim}")
  print(f"The no of rnn in action predictor is {a_rnn} and action hidden dimentsion {h_dimA}")



  gen_config={ 'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,'z_dim':latent_dim,'batch_size': batch_size,
              'h_dim':h_dim, 'past_TS':past_TS, 'future_TS':future_TS, 'is_train':G_train,'hidden_dim_pred':hidden_dim_pred,'hidden_dim_prior':hidden_dim_prior,
               'alpha_dim': alpha_dim,'a_dim': a_dim,'h_dimA':h_dimA,'a_rnn': a_rnn, 'hidden_dim_priorA':hidden_dim_priorA }
  
  
    
    
    
   
  
  if G_train==True or C_train==True:
    prefix = ("{}_{}".format(dataset,model_name)
        + "_image_w="+str(image_w)
        + "_K="+str(past_TS)
        + "_T="+str(future_TS)
        + "_batch_size="+str(global_batch_size)
        + "_alpha="+str(alpha)
        + "_beta="+str(beta)
        + "_A_beta="+str(A_beta)
        + "_lr="+str(lr)+"_z-dim="+str(latent_dim)+"_hidden-dim_pred="+str(hidden_dim_pred)+"_a_rnn="+str(a_rnn)
        +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))

    checkpoint_dir_G = "../models/"+prefix+"/Generator"
    # checkpoint_dir_P = "../models/"+prefix+"/Posterior"
    # checkpoint_dir_C = "../models/"+prefix+"/Critic"
    samples_dir = "../samples/"+prefix+"/"
    logs_dir = "../logs/"+prefix+"/"
    if not exists(checkpoint_dir_G):
        makedirs(checkpoint_dir_G)
    # if not exists(checkpoint_dir_P):
    #   makedirs(checkpoint_dir_P)
    # if not exists(checkpoint_dir_C):
    #     makedirs(checkpoint_dir_C)
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(logs_dir):
        makedirs(logs_dir)
    batched_data, time_steps, val_data=D.load_train_data_w_Val(strategy, dataset, global_batch_size)
    # if reload_ckpt:
    #   epochs=epochs+100

    train_config={'tf_enable':tf_enable, 'alpha':alpha, 'beta':beta, 'epochs':epochs, 'G_train':G_train,'C_train':C_train,'filters':filters, 'ckpt_G': ckpt_G, 
              'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': future_TS, 'margin': margin,'batch_size': batch_size,'c_dim':c_dim,'a_dim':a_dim,
              'checkpoint_dir_G':checkpoint_dir_G,'samples_dir': samples_dir, 'lr':lr, 'beta1':beta1,'A_beta':A_beta, 'gamma': gamma,
              'global_batch_size': global_batch_size, 'time_steps': time_steps, 'reload_ckpt':reload_ckpt,'skip':skip}

    
    if G_train==True and C_train==False:
      print('Training Generator...')
      import train_causal_vleap as Tr
      # import train_vleap as Tr
      # import train_svg as Tr
      train_config['ckpt_G']=ckpt_G
      train_obj=Tr.Train(gen_config,train_config,strategy)
      with strategy.scope():
      
       json_file = open("../models/"+prefix+"/training_config.json","w")
       train_json = json.dumps(train_config, indent=2)
       json_file.write(train_json)
       json_file.close()
       json_gen_file = open("../models/"+prefix+"/training_Generator_config.json","w")
       train_gen_json = json.dumps(gen_config, indent=2)
       json_gen_file.write(train_gen_json)
       json_gen_file.close()
       train_obj.custom_loop(batched_data,val_data)


    
    



if __name__ == '__main__':
  config.define_network_flags()
  app.run(run_main)
