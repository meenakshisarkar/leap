#Code developed by Meenakshi Sarkar, IISc for the project pgvg
import tensorflow as tf
tf.random.set_seed(77)
import os
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow import keras as K

import layers as L

def reparameterize( mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean 

class vgg_cnn(tf.keras.Model):
    def __init__(self,cout,name,**kwargs):
            super().__init__(name=name, **kwargs )
            self.model = K.Sequential([
                L.conv2d(filters=cout,kernel_size=[3,3], activation=None,padding="same", name="cnn"),
                L.batchNorm(name="batch_norm"),
                K.layers.LeakyReLU(alpha=0.2, name="lrelu"),])
    def call(self,input):
        return self.model(input)
    
class vgg_encoder(tf.keras.Model):
    def __init__(self,h_dim,name="vgg_encoder",**kwargs):
        super().__init__(name=name, **kwargs )
        self.h_dim=h_dim
        #64*64
        self.conv0=K.Sequential([
            vgg_cnn(64,name='conv0_0'),
            vgg_cnn(64,name='conv0_1')])
        #32*32
        self.conv1=K.Sequential([
            vgg_cnn(128,name='conv1_0'),
            vgg_cnn(128,name='conv1_1')])
        #16*16
        self.conv2=K.Sequential([
            vgg_cnn(256,name='conv2_0'),
            vgg_cnn(256,name='conv2_1'),
            vgg_cnn(256,name='conv2_2')])
        #8*8
        self.conv3=K.Sequential([
            vgg_cnn(512,name='conv3_0'),
            vgg_cnn(512,name='conv3_1'),
            vgg_cnn(512,name='conv3_2')])
        #4*4
        self.conv4=K.Sequential([
            L.conv2d(filters=self.h_dim,kernel_size=[4,4], activation='tanh',padding="valid", name="conv4_0"),
            L.batchNorm(name="conv4_batch_norm")])
        
        self.max_pool = L.maxpool2D(pool_size=[2, 2],strides=2,name="max_pool")
        self.flatten=L.flatten(name='flatten_emb')

    def call(self,input):
        h0=self.conv0(input)  #64
        h1=self.conv1(self.max_pool(h0)) #64-->32-->32
        h2=self.conv2(self.max_pool(h1)) #32-->16-->16
        h3=self.conv3(self.max_pool(h2)) #16-->8-->8
        h4=self.conv4(self.max_pool(h3)) #8-->4-->1
        h4=self.flatten(h4)
        return  [h4,h0,h1,h2,h3]

class decoder(tf.keras.Model):
    def __init__(self,cdim,h_dim,name="decoder",**kwargs):
        super().__init__(name=name, **kwargs )
        self.cdim=cdim
        self.h_dim=h_dim
        #1 x 1 --> 4 x 4
        self.up_conv0=K.Sequential([
            L.deconv2d(filters=512, kernel_size=[4,4],activation=None,padding='valid', name="up0_conv"),
            L.batchNorm(name="up0_batch_norm"),
            K.layers.LeakyReLU(alpha=0.2, name="up0_lrelu"),])
        #8*8
        self.up_conv1=K.Sequential([
            vgg_cnn(512,name='up_conv1_0'),
            vgg_cnn(512,name='up_conv1_1'),
            vgg_cnn(256,name='up_conv1_2')])
        #16 x16
        self.up_conv2=K.Sequential([
            vgg_cnn(256,name='up_conv2_0'),
            vgg_cnn(256,name='up_conv2_1'),
            vgg_cnn(128,name='up_conv2_2')])
        #32 x32
        self.up_conv3=K.Sequential([
            vgg_cnn(128,name='up_conv3_0'),
            vgg_cnn(64,name='up_conv3_1')])
        #64 x 64
        if self.cdim==3:
         self.up_conv4=K.Sequential([
            vgg_cnn(64,name='up_conv4_0'),
            # L.deconv2d(filters=self.cdim,activation="sigmoid", kernel_size=[3,3],padding='same', name="up4_deconv")]) #for the _new models
            L.deconv2d(filters=self.cdim,activation="tanh", kernel_size=[3,3],padding='same', name="up4_deconv")])
        else:
         self.up_conv4=K.Sequential([
            vgg_cnn(64,name='up_conv4_0'),
            # L.deconv2d(filters=self.cdim,activation="tanh", kernel_size=[3,3],padding='same', name="up4_deconv")])
            L.deconv2d(filters=self.cdim,activation="sigmoid", kernel_size=[3,3],padding='same', name="up4_deconv")]) #svg paper
        # self.up_conv4=K.Sequential([
        #     vgg_cnn(64,name='up_conv4_0'),
        #     L.deconv2d(filters=self.cdim,activation="tanh", kernel_size=[1,1],padding='same', name="up4_deconv")])
        self.ups=L.upsampling2D(size=[2,2],interpolation='nearest')

        self.reshape=K.layers.Reshape((1,1,self.h_dim))

    def call(self, input):
        h_hat=input[0]
        skip=input[1]
        d0=self.up_conv0(self.reshape(h_hat)) #1-->4
        up0=self.ups(d0)  #4-->8
        # d1=self.up_conv1(tf.concat([up0,input[4]],axis=3)) #8x8
        d1=self.up_conv1(tf.concat([up0,skip[3]],axis=3)) #8x8
        up1=self.ups(d1) #8-->16
        # d2=self.up_conv2(tf.concat([up1,input[3]],axis=3)) #16x16
        d2=self.up_conv2(tf.concat([up1,skip[2]],axis=3)) #16x16
        up2=self.ups(d2) #16-->32
        # d3=self.up_conv3(tf.concat([up2,input[2]],axis=3)) #32x32
        d3=self.up_conv3(tf.concat([up2,skip[1]],axis=3)) #32x32
        up3=self.ups(d3) #32-->64
        # output=self.up_conv4(tf.concat([up3,input[1]],axis=3)) #64x64
        output=self.up_conv4(tf.concat([up3,skip[0]],axis=3)) #64x64
        return output

class prediction_rnn(tf.keras.Model):
    def __init__(self,h_dim,hidden_dim,batch_size,name="prediction_rnn",**kwargs):
        super().__init__(name=name, **kwargs )
        self.h_dim=h_dim
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.linear0=L.dense(output_shape=hidden_dim, name='pred_linear0')
        self.lstm0=L.LSTMcell(units=hidden_dim,activation=None,name='prediction_lstm0')
        self.lstm1=L.LSTMcell(units=hidden_dim,activation=None,name='prediction_lstm1')
        self.linear1=L.dense(output_shape=self.h_dim,activation='tanh', name='pred_linear1') ##svg paper
        # self.linear1=L.dense(output_shape=self.h_dim,activation=None, name='pred_linear1')
        # self.init_hidden_state()

    def init_hidden_state(self):
        st0=tf.zeros([self.batch_size,self.hidden_dim])
        self.hidden_st0=[st0,st0]
        self.hidden_st1=[st0,st0]
        return
    def call(self,input):
        h_embd0=self.linear0(tf.concat([input[0],input[1]], axis=1))
        h_embd1,self.hidden_st0 =self.lstm0(h_embd0,self.hidden_st0)
        h_embd2,self.hidden_st1 =self.lstm1(h_embd1,self.hidden_st1)
        h_hat=self.linear1(h_embd2)
        return h_hat
    
class stochastic_rnn(tf.keras.Model):
    def __init__(self,z_dim,hidden_dim,batch_size,name="stochastic_rnn",**kwargs):
        super().__init__(name=name, **kwargs )
        self.z_dim=z_dim
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.linear0=L.dense(output_shape=hidden_dim, name='sto_linear0')
        # self.lstm0=L.LSTMcell(units=hidden_dim,activation='tanh',name='stochastic_lstm0') ## old models not yielding good results
        self.lstm0=L.LSTMcell(units=hidden_dim,activation=None,name='stochastic_lstm0')  ##svg
        self.mu_layer=L.dense(output_shape=self.z_dim,activation=None, name='sto_mu')
        self.logvar_layer=L.dense(output_shape=self.z_dim,activation=None, name='sto_logvar')
        # self.relu=tf.keras.layers.ReLU(max_value=0.1,threshold=0.00, name='stochastic_relu')
        # self.relu=tf.keras.layers.ReLU( name='stochastic_relu')
        # self.init_hidden_state()

    def init_hidden_state(self):
        st0=tf.zeros([self.batch_size,self.hidden_dim])
        self.hidden_st0=[st0,st0]
        return
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    def call(self,input):
        h_embd0=self.linear0(input)
        h_embd1,self.hidden_st0 =self.lstm0(h_embd0,self.hidden_st0)
        mu=self.mu_layer(h_embd1)
        # mu=self.relu(mu)
        logvar=self.logvar_layer(h_embd1)
        z_t= self.reparameterize(mu,logvar)
        return [mu, logvar,z_t]



    