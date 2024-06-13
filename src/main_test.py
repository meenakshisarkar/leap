#Code developed by Meenakshi Sarkar, IISc for the project pgvg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl import app
import tensorflow as tf
tf.random.set_seed(77)
import utils as U

import config 
# from models import Generator
# from generator_v1_PE import Generator_post
# from acvg import acvg
import data.dataloader as D
# import train as Tr
from os.path import exists
from os import makedirs
import json
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12024)])
		
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
	print("I ma here")
		
	strategy = tf.distribute.MirroredStrategy(["GPU:1"])
	# strategy=None
	# global_batch_size = strategy.num_replicas_in_sync*batch_size
	global_batch_size = 128
	# global_batch_size=batch_size

	print(f'Number of devices: {strategy.num_replicas_in_sync}')

	# if not exists(datapath):
	#           raise Exception("Sorry, invalid datapath")

	prefix = ("{}_{}".format(dataset,model_name)
        + "_image_w="+str(image_w)
        + "_K="+str(past_TS)
        + "_T="+str(future_TS)
        + "_batch_size="+str(batch_size)
        + "_alpha="+str(alpha)
        + "_beta="+str(beta)
        + "_lr="+str(lr)+"_z-dim="+str(latent_dim)+"_hidden-dim_pred="+str(hidden_dim_pred)
        +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))
	checkpoint_dir_G = "../models/"+prefix+"/Generator"
	checkpoint_dir_P = "../models/"+prefix+"/Posterior"
	checkpoint_dir_C = "../models/"+prefix+"/Critic"
	# test_data, test_steps= D.load_test_data(strategy,dataname='kth_test',batch= global_batch_size)
	test_data, test_steps= D.load_test_data(strategy,dataname=dataset+'_test',batch= global_batch_size)


	gen_config={ 'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,'z_dim':latent_dim,'batch_size': global_batch_size,
              'h_dim':h_dim, 'past_TS':past_TS, 'future_TS':future_TS, 'is_train':G_train,'hidden_dim_pred':hidden_dim_pred, 'hidden_dim_prior':hidden_dim_prior}

	critic_config={ 'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,
							'filters':filters, 'past_TS':past_TS, 'future_TS':future_TS, 'is_train':C_train}


	pgvg_config={'image_h':image_h, 'image_w':image_w,'c_dim':c_dim, 'C_train':C_train,
							'filters':filters, 'past_TS':past_TS, 'future_TS':future_TS, 'G_train':G_train}

	test_config={'tf_enable':tf_enable, 'is_train':G_train,'C_train':C_train,'filters':filters,'batch_size':global_batch_size, 
	              'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': test_future_TS, 'model_name': model_name, 'ckpt_G': ckpt_G, 'h_dim':h_dim,'z_dim':latent_dim,
	              'checkpoint_dir_G':checkpoint_dir_G,'c_dim':c_dim,'a_dim':a_dim,'dataset':dataset,
								'global_batch_size': global_batch_size, 'time_steps': test_steps, 'n_samples': n_samples,'skip':skip}
	# with tf.device("/gpu:{}".format(gpu)):
		
	if G_train==True or C_train== True:
		raise Exception("Sorry, Model cannot be tested with training flags for generator or critic as true. \n Please set G_train and C_train= False in setup.bash")
	else:
		if test_G==True:
				# import test_lfvg as Ts
				# import test_lfvg_fs as Ts
				# import test_lfvg_fs_alt as Ts
				# import test_lfvg_fs_gray as Ts
				# import test_lfvg_fs_gray_vgg as Ts
				# import test_lfvg_gray as Ts
				# import test_lfvg_gray_vgg as Ts
				# import test_lfvg_conv as Ts
				# import test_svg_vgg as Ts
				import test_svg as Ts
				model_name=f'{model_name}_Generator'
				print("Testing {}....".format(model_name))
				test_obj=Ts.Test(gen_config,test_config,strategy)
	#   	

		results_prefix= ("{}_{}".format(dataset,test_config['model_name'])
	    #   + "_GPU_id="+str(gpu)
	      + "_image_w="+str(image_w)
	      + "_K="+str(past_TS)
	      + "_T="+str(test_future_TS)
	      + "_batch_size="+str(batch_size)
	      + "_alpha="+str(alpha)
	      + "_beta="+str(beta)
	      + "_lr="+str(lr)+"_z-dim="+str(latent_dim)+"_hidden-dim_pred="+str(hidden_dim_pred)
          +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))
		with strategy.scope():
			test_config['results_prefix']=results_prefix
			test_obj.custom_loop(test_data)
			test_json = json.dumps(test_config)
			json_file = open("../results/quantitative/"+dataset+"/"+results_prefix+"/testing_config.json","w")
			json_file.write(test_json)
			json_file.close()
			



if __name__ == '__main__':
	config.define_network_flags()
	app.run(run_main)
