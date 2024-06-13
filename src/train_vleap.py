#Code developed by Meenakshi Sarkar, IISc for the project pgvg
import tensorflow as tf
tf.random.set_seed(77)
import sys
# tf.compat.v1.enable_eager_execution()
# from joblib import Parallel, delayed
import time
import utils as U
import numpy as np
np.random.seed(77)
import loss as Ls
# from tensorflow import keras
from tensorflow.keras.metrics import Mean
# from models import Generator
from generator_vgleap import Generator as Generator
import pickle
import data.dataloader as D
import cv2 as cv2
from tqdm import trange,tqdm
# from config import Config as cfg
# import tensorflow.contrib.eager as tfe

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    if percent_done==100:
      print('[%s] %f%s\r' % (bar, percent_done, '%'))

class Train(object):
  def __init__(self,gen_config,cfg, strategy: tf.distribute.Strategy):
    self.strategy = strategy
    self.cfg=cfg
    self.global_batch_size=self.cfg['global_batch_size']

    with self.strategy.scope():
     
     self.gen=Generator(gen_config,skip=self.cfg["skip"],name="vleap")
     self.optimizer_gen= tf.keras.optimizers.Adam(learning_rate=self.cfg['lr'], beta_1=self.cfg['beta1'])
     print("Training Variational LeAP  model with Sarkar et al NeurIPS paper config")
    
    #  self.gen.build(input_shape=[(None,self.cfg['past_TS']+self.cfg['future_TS'],self.cfg['image_h'], self.cfg['image_w'],self.cfg['c_dim']),
    #                              (None,self.cfg['past_TS']+self.cfg['future_TS'],self.cfg['a_dim'])])
  
    
    #  self.gen.summary(expand_nested=True)
    

     self.checkpoint_gen = tf.train.Checkpoint(optimizer=self.optimizer_gen, model=self.gen)
     self.manager_gen = tf.train.CheckpointManager(self.checkpoint_gen, directory=self.cfg['checkpoint_dir_G'], max_to_keep=200)
     if self.cfg['reload_ckpt']:
      checkpoint_path_gen=self.cfg['checkpoint_dir_G']+'/ckpt-'+self.cfg['ckpt_G']  #63'
      restore_gen=self.checkpoint_gen.restore(checkpoint_path_gen).expect_partial()
      if restore_gen:
       print("Restored from {}".format(checkpoint_path_gen))
      else:
       raise Exception("No valid check point for generator found")
    
     self.recon_loss= Ls.recon_loss_l1(batch_size=self.global_batch_size,reduction=tf.keras.losses.Reduction.NONE)
     self.kld_loss=Ls.KL_Divergence(batch_size=self.global_batch_size,reduction=tf.keras.losses.Reduction.NONE)
     self.action_loss= Ls.recon_loss_l2(batch_size=self.global_batch_size,reduction=tf.keras.losses.Reduction.NONE)
      
      
     self.L_recon_metric= Mean(name="reconst_loss")
     self.kld_metric=Mean(name="KL_div")
     self.L_action_metric= Mean(name="action_loss")
                      
  def compute_loss(self,input):    # input is a dictionary of predict,gen_vel_map, gt,gt_vel_map,D_real,D_logits_real,D_fake, D_logits_fake )
      # replica_context = tf.distribute.get_replica_context()  # for strategy
      # print("in compute loss context is {}".format(replica_context))
      # shape=input['gt']
      # print("shape={}".format(shape))
      Loss_p0=self.recon_loss(input['predict'],input['gt'])
      # gdl=self.gradient_loss(input['predict'],input['gt'])
      # vgdl=self.gradient_loss(input['gen_vel_map'], input['gt_vel_map'])
      kld=self.kld_loss(mu_pred=input['post_mean'],logvar_pred=input['post_logvar'],
                        mu_true=input['prior_mean'],logvar_true=input['prior_logvar'])
      reconst_loss= Loss_p0
      action_loss=self.action_loss(input['at_predict'],input['at_gt'])
      # final_loss +=tf.expand_dims(self.cfg['beta']*tf.nn.compute_average_loss(tf.expand_dims(kld,axis=-1),
      #                                                                         global_batch_size=self.global_batch_size),axis=-1)
      final_loss=tf.nn.compute_average_loss(self.cfg['alpha']*reconst_loss+self.cfg['beta']*kld+self.cfg['A_beta']*action_loss)
      loss={  "KL_div":kld, 'reconst_loss': reconst_loss, 'final_loss': final_loss, 'action_loss': action_loss }
            
      return loss
   
  #  @tf.function
  def train_step(self, input): 
      
      stidx = np.random.randint(0, 10)
      img_in=input[0]
      # img_in=img_in[...,:3]/255
      act_in=(input[1]+1.0)/2
      img_in=D.normalize_image(img_in[...,:3])
      im_seq=img_in[:,stidx:self.cfg['past_TS']+self.cfg['future_TS']+stidx,:,:,:]
      target=im_seq[:,1:self.cfg['past_TS']+self.cfg['future_TS'],...]

      at_seq=tf.squeeze(act_in[:,stidx:self.cfg['past_TS']+self.cfg['future_TS']+stidx,:])

      target_at=at_seq[:,1:self.cfg['past_TS']+self.cfg['future_TS'],...]
      
  
      
      with tf.GradientTape() as tape0 :
        output = self.gen([im_seq,at_seq], training=self.cfg['G_train'])
        predict= output[0]
        # predict=tf.repeat(predict, repeats=[3], axis=4)
        posterior_mean=output[1]
        posterior_logvar=output[2]
        prior_mean=output[3]
        prior_logvar=output[4]
        at_predict=output[5]
        
        
        model_output_dict={'predict': predict,'gt':target ,
                  "post_mean":posterior_mean , 'post_logvar':posterior_logvar,
                  "prior_mean":prior_mean, 'prior_logvar':prior_logvar, 
                  "at_predict":at_predict, "at_gt": target_at}
        
        loss_dict=self.compute_loss(model_output_dict)
        
      if self.cfg['G_train']==True:
        self.gen_vars = self.gen.trainable_variables  
        # tape0.watch(self.gen_vars) 
        grads_gen = tape0.gradient(loss_dict['final_loss'], self.gen_vars)
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.gen_vars))
      self.update_metrics(loss_dict)                 
      ouptput_dict={'predict': predict[0,:,:,:,:],'gt':target[0,:,:,:,:],'loss_dict': loss_dict,
                    'at_predict': at_predict,'at_gt':target_at} 

      return ouptput_dict
   
  def val_step(self, input): ## input is a disctionary having input of in_seq,xt,vel_seq,action_past_seq, target): 
    stidx = np.random.randint(0, 10)
    img_in=input[0]
    # img_in=img_in[...,:3]/255
    act_in=(input[1]+1.0)/2 
    # input=input[...,:3]/255
    img_in=D.normalize_image(img_in[...,:3])
    im_seq=img_in[:,stidx:self.cfg['past_TS']+self.cfg['future_TS']+stidx,:,:,:]
    target=im_seq[:,self.cfg['past_TS']:self.cfg['past_TS']+self.cfg['future_TS'],...]
    at_seq=tf.squeeze(act_in[:,stidx:self.cfg['past_TS']+self.cfg['future_TS']+stidx,:])
    target_at=at_seq[:,self.cfg['past_TS']:self.cfg['past_TS']+self.cfg['future_TS'],...]

    predict=[]
    at_predict=[]
    prior_mean=[]
    prior_logvar=[]
    posterior_mean=[]
    posterior_logvar=[]
    current_img=im_seq[:,0,...]
    current_at=at_seq[:,0,:]
    self.gen.prediction_rnn.init_hidden_state()
    self.gen.prior_rnn.init_hidden_state()
    self.gen.posterior_rnn.init_hidden_state()
    self.gen.at_predictor.init_hidden_state()

    for step in range(self.cfg['past_TS']+self.cfg['future_TS']-1):
      h_embd=self.gen.img_encoder(current_img,training=self.cfg['G_train'])
      h_t=h_embd[0]
      a_embd=self.gen.at_encoder(current_at,training=self.cfg['G_train'])
      if self.cfg['skip']==True or step<self.cfg['past_TS']:
        skip=[h_embd[1],h_embd[2],h_embd[3],h_embd[4]]

      prior_z_t=self.gen.prior_rnn(tf.concat([h_t,a_embd],axis=1),training=self.cfg['G_train'])
      prior_mu_t=prior_z_t[0]
      prior_logvar_t=prior_z_t[1]
      z_t=prior_z_t[2]
        # z_t=prior_mu_t
      h_target_embd=self.gen.img_encoder(im_seq[:,step+1,...],training=self.cfg['G_train'])
      h_target=h_target_embd[0]
      future_at=self.gen.at_encoder(at_seq[:,step+1,...],training=self.cfg['G_train'])
      # current_flow=im_seq[:,step+1,...]-current_img
      z=self.gen.posterior_rnn(tf.concat([h_target,future_at],axis=1),training=self.cfg['G_train'])
      mu_t=z[0]
      logvar_t=z[1]
      # z_t=z[2]
      h_next=self.gen.prediction_rnn([h_t,z_t],training=self.cfg['G_train'])
      a_hat_next=self.gen.at_predictor([a_embd,z_t],training=self.cfg['G_train'])
      # x_next=self.decoder([h_next,h_embd[1],h_embd[2],h_embd[3],h_embd[4]],training=self.cfg['future_TS']self.cfg['G_train'])
      
      if step<self.cfg['past_TS']-1:
        current_img=im_seq[:,step+1,...]
        current_at=at_seq[:,step+1,...]
        x_next=current_img
        a_next=current_at
      else:
        x_next=self.gen.decoder([h_next,skip],training=self.cfg['G_train'])
        # x_tilde=self.gen.decoder([h_next,skip,current_img],training=self.cfg['G_train']) #for generator_lfvg_mod
        # x_next=x_tilde
        a_next=self.gen.at_decoder(a_hat_next,training=self.cfg['G_train'])
        current_img=x_next
        current_at=a_next
        # if self.cfg['c_dim']==1:
        #  current_img=tf.image.grayscale_to_rgb(current_img)
        predict.append(x_next)
        at_predict.append(a_next)
        posterior_mean.append(mu_t)
        posterior_logvar.append(logvar_t)
        prior_mean.append(prior_mu_t)
        prior_logvar.append(prior_logvar_t)
        
    predict=tf.stack(predict,axis=1)
    at_predict=tf.stack(at_predict,axis=1)
    prior_mean=tf.stack(prior_mean,axis=1)
    prior_logvar=tf.stack(prior_logvar,axis=1)
    posterior_mean=tf.stack(posterior_mean,axis=1)
    posterior_logvar=tf.stack(posterior_logvar,axis=1)

    model_output_dict={'predict': predict,'gt':target ,
                "post_mean":posterior_mean , 'post_logvar':posterior_logvar,
                "prior_mean":prior_mean, 'prior_logvar':prior_logvar,
                "at_predict":at_predict, "at_gt": target_at}
    loss_dict=self.compute_loss(model_output_dict)  
    self.update_metrics(loss_dict)            
    ouptput_dict={'predict': predict[0,:,:,:,:],'gt':target[0,:,:,:,:] ,
                 'at_predict': at_predict[0,:,:],'at_gt':target_at[0,:,:],'loss_dict': loss_dict} 

    return ouptput_dict
  def reset_metrics(self):
      self.L_recon_metric.reset_state()
      self.kld_metric.reset_state() 
      self.L_action_metric.reset_state()
  def update_metrics(self,d):
    self.L_recon_metric.update_state(d['reconst_loss'])
    self.kld_metric.update_state(d["KL_div"])
    self.L_action_metric.update_state(d["action_loss"])

  def reduce_dict(self, d: dict):
    """ reduce items in dictionary d """
    recon_loss=self.strategy.reduce(tf.distribute.ReduceOp.SUM, d['reconst_loss'], axis=None) 
    kld=self.strategy.reduce(tf.distribute.ReduceOp.SUM, d["KL_div"], axis=None)
    final_loss=self.strategy.reduce(tf.distribute.ReduceOp.SUM, d["final_loss"], axis=None)
    action_loss=self.strategy.reduce(tf.distribute.ReduceOp.SUM, d["action_loss"], axis=None)
    return [recon_loss,kld,final_loss,action_loss]
  @tf.function
  def distributed_train_step(self, data):
    results = self.strategy.run(self.train_step, args=(data,))
    # print("I am in distributed train")
    loss=self.reduce_dict(results['loss_dict'])
    return {'predict':results['predict'], 'target':results['gt'],'loss':loss,
            'at_predict':results['at_predict'], 'target_at':results['at_gt']}
   
  @tf.function
  def distributed_val_step(self, data):
      results = self.strategy.run(self.val_step, args=(data,))
      # print("I got in here")
      loss=self.reduce_dict(results['loss_dict'])
      return {'predict':results['predict'], 'target':results['gt'],'loss':loss,
              'at_predict':results['at_predict'], 'target_at':results['at_gt']}
  def custom_loop(self,batched_data,val_data):
    self.tf_enable=self.cfg['tf_enable']
    epochs = self.cfg['epochs']
    total_itr=600
    if self.cfg['reload_ckpt']:
      ckpt=int(self.cfg['ckpt_G'])
    #  epochs=epochs-ckpt
    else:
      ckpt=0


    past_TS=self.cfg['past_TS']
    # time_steps=self.cfg['time_steps']
    print(self.global_batch_size)
    # with self.strategy.scope():
    data_iterator=iter(batched_data)
    val_iterator=iter(val_data)
    for epoch in range(ckpt,epochs):
      step=0
      itr=1
      recon_loss=0
      kld_loss=0
      final_loss=0
      action_loss=0
      # for data in batched_data:
      for itr in tqdm(range(total_itr)): #for local server
      # for itr in range(total_itr):  #for dgx
        data=next(data_iterator)
        # data=batched_data()
        self.cfg['G_train']=True
        images=data['video']
        actions=data['action']
        # self.cfg['G_train']=True
        # images=data['image']
        model_output=self.distributed_train_step([images,actions])
        losses=model_output['loss']
        recon_loss=recon_loss+self.L_recon_metric.result()
        kld_loss=kld_loss+self.kld_metric.result()
        final_loss=final_loss+losses[2]
        action_loss=action_loss+self.L_action_metric.result()
        itr=itr+1
        
        # progress_bar(( step+ 1) / (time_steps) * 100, 60)
        step=step+1                        
      _samples = model_output['predict'].values #for multi-gpu
      samples=_samples[0] #for multi-gpu
      _sbatch = model_output['target'].values #for multi-gpu
      sbatch=_sbatch[0] #for multi-gpu
      samples = np.concatenate((samples, sbatch), axis=0)
      # print("Saving sample ...")
      U.save_images(samples[:, :, :, :], [2, self.cfg['past_TS']+self.cfg['future_TS']-1],
                  self.cfg['samples_dir']+"train_%s.png" % (epoch))
      # U.imsave(samples[:, :, :, :]*255, [2, self.cfg['past_TS']+self.cfg['future_TS']-1],
      #             self.cfg['samples_dir']+"train_%s.png" % (epoch))'''
      
      # with open('optimizer.pkl', 'rb') as f:
      #   store_state = pickle.load(f)
      # self.optimizer_gen._load_own_variables(store_state)
      # print("training done for epoch: {}".format(epoch+ckpt))
      print("training done for epoch: {}".format(epoch))
      template = ('Epoch: {}, Reconstruction Loss: {}, '
              'KL_divergence: {}, Final_loss: {}, Action_loss: {}')
      
      print(template.format(epoch, recon_loss*1.0/itr,
                                kld_loss*1.0/itr,final_loss*1.0/itr,action_loss*1.0/itr))

      if epoch != self.cfg['epochs'] - 1:
        self.reset_metrics()
      print("I have val data")
      
      val_input=next(val_iterator)
      val_images=val_input['video']
      val_actions=val_input['action']
      # val_folder_len=val_input['length']
      self.cfg['G_train']=False
      # val_model_output=self.distributed_val_step([val_images,val_folder_len])
      val_model_output=self.distributed_val_step([val_images,val_actions])
      '''if epoch%1==0:
        for data in val_data:
          print("I have val data")
          val_images=data['image']
          self.cfg['G_train']=False
          val_model_output=self.distributed_val_step(val_images)'''
          
          
      template = ('Training Epoch: {}, Reconstruction Loss: {}, '
            'KL_divergence: {}, Action_loss: {}')
      print(template.format(epoch, self.L_recon_metric.result(),
                                  self.kld_metric.result(),self.L_action_metric.result()))
          
          
      _samples = val_model_output['predict'].values #for multi-gpu
      samples=_samples[0] #for multi-gpu
      # samples=_samples[0] #for multi-gpu
      _sbatch = val_model_output['target'].values #for multi-gpu
      sbatch=_sbatch[0] #for multi-gpu'''
        
      '''_samples = val_model_output['predict'] #for single-gpu
      samples=np.asarray(tf.image.grayscale_to_rgb(_samples)*255) #for single-gpu
      # samples=_samples[0] #for multi-gpu
      # samples=_samples
      # samples=Image.fromarray((samples).astype(np.uint8))
      _sbatch = val_model_output['target'] #for single-gpu
      sbatch=np.asarray(tf.image.grayscale_to_rgb(_sbatch)*255) #for single-gpu'''

      print("shape of target {}".format(samples.shape))
      print("shape of pred_data {}".format(sbatch.shape))
      # print("shape of in_seq_data {}".format(in_seq.shape))
      # _samples = model_output['predict']
      # samples=_samples
      # _sbatch = model_output['target']
      # sbatch=_sbatch
      Samples = np.concatenate((samples[...], sbatch[...]), axis=0)
      # Samples = np.concatenate((samples[self.cfg['past_TS']-2:,...], sbatch[self.cfg['past_TS']-2:,...]), axis=0)
      # Samples=Samples.astype(np.unit8)
      print("shape of Samples {}".format(Samples.shape))
      print("Saving sample ...")
      U.save_images(Samples,[2, self.cfg['future_TS']], self.cfg['samples_dir']+"train_new_%s.png" % (epoch))
      # U.imsave(Samples*255,[2, self.cfg['future_TS']], self.cfg['samples_dir']+"train_new_%s.png" % (epoch))'''
      

      self.manager_gen.save()
        

      # if epoch%100==99:
      #    tf.saved_model.save(self.gen,self.cfg['checkpoint_dir_G'])
    return

          

          
         


      

