#Code developed by Meenakshi Sarkar, IISc for the project pgvg
import tensorflow as tf
tf.random.set_seed(77)
import os
from tensorflow import keras
from tensorflow import keras as K
import layers as L
from models_svg import *
import models_actor as A

class Generator(tf.keras.Model):
  def __init__(self, cfg,skip=False,name="vleap",**kwargs):
    super().__init__(name=name,**kwargs)
    self.cfg=cfg
    self.skip=skip
    self.img_encoder=vgg_encoder(h_dim=self.cfg['h_dim'],name="vgg_encoder")
    self.at_encoder=A.ac_encoder(h_dimA=self.cfg['h_dimA'],name="act_encoder")
    self.decoder=decoder(h_dim=self.cfg['h_dim'],cdim=self.cfg['c_dim'],name="decoder")
    self.at_decoder=A.ac_decoder(a_dim=self.cfg['a_dim'],name="act_decoder")
    self.prediction_rnn=prediction_rnn(h_dim=self.cfg['h_dim'],hidden_dim=self.cfg['hidden_dim_pred'],
             batch_size=self.cfg['batch_size'],name="prediction_rnn")

    self.at_predictor=A.ac_predictor_causal(h_dimA=self.cfg['h_dimA'],a_rnn=self.cfg['a_rnn'],
             batch_size=self.cfg['batch_size'],name="at_predictor")
    # self.prediction_rnn.init_hidden_state()
    self.prior_rnn=stochastic_rnn(z_dim=self.cfg['z_dim'],hidden_dim=self.cfg['hidden_dim_prior'],
             batch_size=self.cfg['batch_size'],name="prior_rnn")
    # self.prior_rnn.init_hidden_state()
    self.posterior_rnn=stochastic_rnn(z_dim=self.cfg['z_dim'],hidden_dim=self.cfg['hidden_dim_prior'],
             batch_size=self.cfg['batch_size'],name="posterior_rnn")
    
    self.A_prior_rnn=A.ac_stochastic(alpha_dim=self.cfg['alpha_dim'],rnn_dim=self.cfg['hidden_dim_priorA'],
             batch_size=self.cfg['batch_size'],name="at_prior_rnn")
    # self.prior_rnn.init_hidden_state()
    self.A_posterior_rnn=A.ac_stochastic(alpha_dim=self.cfg['alpha_dim'],rnn_dim=self.cfg['hidden_dim_priorA'],
             batch_size=self.cfg['batch_size'],name="at_posterior_rnn")

    im_seq_shape = [self.cfg['past_TS']+self.cfg['future_TS'],self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]
    at_seq_shape = [self.cfg['past_TS']+self.cfg['future_TS'], self.cfg['a_dim']]
    # target_vel_shape = [self.cfg['future_TS'], self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]
    self.im_seq= L.input(input_shape=im_seq_shape,batch_size=None, name="xt")
    self.at_seq= L.input(input_shape=at_seq_shape,batch_size=None, name="at")
    # self.target_vel_seq= L.input(input_shape=target_vel_shape,batch_size=None, name="vel_seq")
            

  def call(self,input):
    # _shape=_action_seq.shape.as_list()
    im_seq=self.im_seq(input[0])
    at_seq=self.at_seq(input[1])
    # target_vel_seq=self.target_vel_seq(input[1]) 
    self.prediction_rnn.init_hidden_state()
    self.prior_rnn.init_hidden_state()
    self.posterior_rnn.init_hidden_state()
    self.at_predictor.init_hidden_state()
    self.A_prior_rnn.init_hidden_state()
    self.A_posterior_rnn.init_hidden_state()
    prior_mean=[]
    prior_logvar=[]
    posterior_mean=[]
    posterior_logvar=[]
    Ac_prior_mean=[]
    Ac_prior_logvar=[]
    Ac_posterior_mean=[]
    Ac_posterior_logvar=[]


    predict = []
    at_predict=[]
    for step in range(self.cfg['past_TS']+self.cfg['future_TS']-1):
      h_embd=self.img_encoder(im_seq[:,step,...],training=self.cfg['is_train'])
      h_t=h_embd[0]
      a_embd=self.at_encoder(at_seq[:,step,...],training=self.cfg['is_train'])
      if self.skip== True or step<self.cfg['past_TS']:
        skip=[h_embd[1],h_embd[2],h_embd[3],h_embd[4]]
      # elif self.skip== False and step<self.cfg['past_TS']:
      #   skip=[h_embd[1],h_embd[2],h_embd[3],h_embd[4]]
          
      prior_zt=self.prior_rnn(h_t,training=self.cfg['is_train'])
      mu_t=prior_zt[0]
      logvar_t=prior_zt[1]
      zt_prior=prior_zt[2]

      prior_at=self.A_prior_rnn([a_embd,zt_prior],training=self.cfg['is_train'])
      alpha_mu=prior_at[0]
      alpha_logvar=prior_at[1]

      future_ht=self.img_encoder(im_seq[:,step+1,...],training=self.cfg['is_train'])
      future_a_embd=self.at_encoder(at_seq[:,step+1,...],training=self.cfg['is_train'])

      post_zt=self.posterior_rnn(future_ht[0],training=self.cfg['is_train'])
      post_mu=post_zt[0]
      post_logvar=post_zt[1]
      z_t=post_zt[2]

      post_at=self.A_posterior_rnn([future_a_embd,z_t],training=self.cfg['is_train'])
      post_alpha_mu=post_at[0]
      post_alpha_logvar=post_at[1]
      alpha_t=post_at[2]

      h_next=self.prediction_rnn([tf.concat([h_t,a_embd],axis=1),z_t],training=self.cfg['is_train'])
      x_next=self.decoder([h_next,skip],training=self.cfg['is_train'])
      a_hat_next=self.at_predictor([a_embd,alpha_t],training=self.cfg['is_train'])
      a_next=self.at_decoder(a_hat_next,training=self.cfg['is_train'])

      predict.append(x_next)
      at_predict.append(a_next)
      prior_mean.append(mu_t)
      prior_logvar.append(logvar_t)
      posterior_mean.append(post_mu)
      posterior_logvar.append(post_logvar)

      Ac_prior_mean.append(alpha_mu)
      Ac_prior_logvar.append(alpha_logvar)
      Ac_posterior_mean.append(post_alpha_mu)
      Ac_posterior_logvar.append(post_alpha_logvar)

    predict=tf.stack(predict,axis=1)
    at_predict=tf.stack(at_predict,axis=1)
    prior_mean=tf.stack(prior_mean,axis=1)
    prior_logvar=tf.stack(prior_logvar,axis=1)
    posterior_mean=tf.stack(posterior_mean,axis=1)
    posterior_logvar=tf.stack(posterior_logvar,axis=1)

    Ac_prior_mean=tf.stack(Ac_prior_mean,axis=1)
    Ac_prior_logvar=tf.stack(Ac_prior_logvar,axis=1)
    Ac_posterior_mean=tf.stack(Ac_posterior_mean,axis=1)
    Ac_posterior_logvar=tf.stack(Ac_posterior_logvar,axis=1)
    return [predict, posterior_mean,posterior_logvar,prior_mean, prior_logvar,
             at_predict,Ac_posterior_mean,Ac_posterior_logvar,Ac_prior_mean, Ac_prior_logvar  ] ##tf.stack might be helpful
