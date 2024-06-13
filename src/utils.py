"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2 as cv2
import random
import imageio
# from skimage.color import rgb2gray
import scipy.misc
import numpy as np
import os
import glob
from PIL import Image
from joblib import Parallel, delayed
np.random.seed(77)
import tensorflow as tf
tf.random.set_seed(77)

SEED= 77
AUTOTUNE = tf.data.AUTOTUNE


def parallel_data(data_batch,past_TS=5,future_TS=10, image_h=64,image_w=64):
   _shape=data_batch[0].shape
  #  print(_shape)
   batch_size=_shape[0]
   with Parallel(n_jobs=batch_size) as parallel:
      output = parallel(delayed(load_roam_data_mod)(data_batch[0][i], data_batch[1][i],(image_h, image_w), past_TS, future_TS) for i in range(batch_size))
      # output = parallel(delayed(load_roam_data_mod_half_fps)(data_batch[0][i], data_batch[1][i],(image_h, image_w), past_TS, future_TS) for i in range(batch_size))
   return output

  # return input
def gen_preprocessed_vid(inputs,past_TS):
  seq_batch=[]
  diff_batch=[]
  action_batch=[]
  for seq in inputs:
    seq_frames = seq[0]
    print("I reached here")
    seq_frames= np.expand_dims(seq_frames, axis=0)
    seq_batch.append(seq_frames)

    diff_frames = seq[1]
    diff_frames= np.expand_dims(diff_frames, axis=0)
    diff_batch.append(diff_frames)

    action_frames = seq[3]
    action_frames= np.expand_dims(action_frames, axis=0)
    action_batch.append(action_frames)
  
  gt_frames= np.concatenate(seq_batch, axis=0)
  vel_seq= np.concatenate(diff_batch, axis=0)
  action_seq= np.concatenate(action_batch, axis=0)
  action_past_seq= action_seq[:,:past_TS,:,:,:]
  at_in_seq=tf.squeeze(action_past_seq[:,:,0,0,:])
  action_future_seq= action_seq[:,past_TS:,:,:,:]
  at_target=tf.squeeze(action_future_seq[:,:,0,0,:])

  xt= gt_frames[:,past_TS-1,...]
  in_seq=gt_frames[:,:past_TS,:,:,:]
  target=gt_frames[:,past_TS:,:,:,:]
  output={'in_seq': in_seq, 'xt': xt,'vel_seq':vel_seq,
          'action_past_seq':action_past_seq, 'target':target,
          'at_in_seq':at_in_seq, 'at_target':at_target}
  return output
def load_roam_dict(data_path,buffer_size,batch_size):
  data_path = data_path
  data_dirs=[]
  dirs_len=[]
  for d1 in os.listdir(data_path):
    for d2 in os.listdir(os.path.join(data_path, d1)):
        date_path=os.path.join(data_path, d1)
        for d3 in os.listdir(os.path.join(date_path, d2)):
            location_path=os.path.join(date_path, d2)
            if d3.split('_')[0]== "Zed":
                rosbag_path=os.path.join(location_path,"Rosbag_"+d3.split('_')[1]+"/")
                with open(rosbag_path+"control_actions.txt") as f: 
                    action_list = f.readlines()
                dir_len=int(len(os.listdir(os.path.join(location_path,d3+"/left/"))))
                for l in range(dir_len//50):
                    if ((l+1)*50)//3 >= len(action_list):
                        break
                    else:   
                        data_dirs.append(os.path.join(location_path,d3+"/left"+"#"+str(l)))
                        dirs_len.append([l*50,l*50+50])
# train_dirs= np.asarray(train_dirs)
# dirs_len= np.asarray(dirs_len)
# data_dict= dict(zip(train_dirs,dirs_len))
  train_dirs_dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
  dirs_len_dataset = tf.data.Dataset.from_tensor_slices(dirs_len)
  dataset=tf.data.Dataset.zip((train_dirs_dataset,dirs_len_dataset))
  batched_dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
  # batched_dataset = strategy.experimental_distribute_dataset(batched_dataset)
  return batched_dataset
def load_roam_dict_stg(data_path,buffer_size,batch_size,strategy):
  data_path = data_path
  data_dirs=[]
  dirs_len=[]
  for d1 in os.listdir(data_path):
    for d2 in os.listdir(os.path.join(data_path, d1)):
        date_path=os.path.join(data_path, d1)
        for d3 in os.listdir(os.path.join(date_path, d2)):
            location_path=os.path.join(date_path, d2)
            if d3.split('_')[0]== "Zed":
                rosbag_path=os.path.join(location_path,"Rosbag_"+d3.split('_')[1]+"/")
                with open(rosbag_path+"control_actions.txt") as f: 
                    action_list = f.readlines()
                dir_len=int(len(os.listdir(os.path.join(location_path,d3+"/left/"))))
                for l in range(dir_len//50):
                    if ((l+1)*50)//3 >= len(action_list):
                        break
                    else:   
                        data_dirs.append(os.path.join(location_path,d3+"/left"+"#"+str(l)))
                        dirs_len.append([l*50,l*50+50])
# train_dirs= np.asarray(train_dirs)
# dirs_len= np.asarray(dirs_len)
# data_dict= dict(zip(train_dirs,dirs_len))
  train_dirs_dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
  dirs_len_dataset = tf.data.Dataset.from_tensor_slices(dirs_len)
  dataset=tf.data.Dataset.zip((train_dirs_dataset,dirs_len_dataset))
  batched_dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
  batched_dataset = strategy.experimental_distribute_dataset(batched_dataset)
  return batched_dataset
def generate_dist_data(filepaths,past_TS=5,future_TS=10, image_h=64,image_w=64,c_dim=3,a_dim=2):
   output=parallel_data(filepaths,past_TS,future_TS,image_h, image_w)
   print("I reached here1")
   _dataset=gen_preprocessed_vid(output,past_TS)
  #  in_seq=_dataset['in_seq']
  #  xt=_dataset['xt']
  #  dataset_img1 = tf.data.Dataset.from_tensor_slices(_dataset['in_seq'],_dataset['xt'])
   dataset_in_seq = tf.data.Dataset.from_tensor_slices(_dataset['in_seq'])
   dataset_xt = tf.data.Dataset.from_tensor_slices(_dataset['xt'])
   dataset_vel_seq = tf.data.Dataset.from_tensor_slices(_dataset['vel_seq'])
   dataset_action_past_seq = tf.data.Dataset.from_tensor_slices(_dataset['action_past_seq'])
   dataset_target = tf.data.Dataset.from_tensor_slices(_dataset['target'])
   dataset_at_in_seq = tf.data.Dataset.from_tensor_slices(_dataset['at_in_seq'])
   dataset_at_target = tf.data.Dataset.from_tensor_slices(_dataset['at_target'])
   dataset_temp1=tf.data.Dataset.zip((dataset_in_seq,dataset_xt))
  #  dataset_temp1=tf.data.Dataset.zip((dataset_in_seq,dataset_xt))
   dataset_temp2=tf.data.Dataset.zip((dataset_vel_seq,dataset_action_past_seq))
   dataset_temp3=tf.data.Dataset.zip((dataset_target,dataset_at_in_seq))
   dataset_temp4=tf.data.Dataset.zip((dataset_temp1,dataset_temp2))
   dataset_temp5=tf.data.Dataset.zip((dataset_temp3,dataset_at_target))
   dataset=tf.data.Dataset.zip((dataset_temp4,dataset_temp5))

   


   
  #  signature_dict={'in_seq': tf.TensorSpec(shape=(None,past_TS,image_h,image_w,c_dim),dtype=tf.float32, name='in_seq'),
  #     'xt': tf.TensorSpec(shape=(None,image_h,image_w,c_dim),dtype=tf.float32, name='xt'),
  #     'vel_seq': tf.TensorSpec(shape=(None,past_TS-1,image_h,image_w,c_dim),dtype=tf.float32, name='vel_seq'),
  #     'action_past_seq': tf.TensorSpec(shape=(None,past_TS,image_h//8,image_w//8,a_dim),dtype=tf.float32, name='action_past_seq'),
  #     'target': tf.TensorSpec(shape=(None,future_TS,image_h,image_w,c_dim),dtype=tf.float32, name='target'),
  #     'at_in_seq': tf.TensorSpec(shape=(None,past_TS,a_dim),dtype=tf.float32, name='at_in_seq'),
  #     'at_target': tf.TensorSpec(shape=(None,future_TS,a_dim),dtype=tf.float32, name='at_target')}
  #  tensor_output=tf.ragged.constant(output)
  #  past_TS=tf.constant(past_TS)
   

   
   
  #  dataset=tf.data.Dataset.from_generator(gen_preprocessed_vid, args=[output,past_TS],output_signature=
  #    [tf.TensorSpec(shape=(None,past_TS,image_h,image_w,c_dim),dtype=tf.float32, name='in_seq'),
  #     tf.TensorSpec(shape=(None,1,image_h,image_w,c_dim),dtype=tf.float32, name='xt'),
  #     tf.TensorSpec(shape=(None,past_TS-1,image_h,image_w,c_dim),dtype=tf.float32, name='vel_seq'),
  #     tf.TensorSpec(shape=(None,past_TS,image_h//8,image_w//8,a_dim),dtype=tf.float32, name='action_past_seq'),
  #     tf.TensorSpec(shape=(None,future_TS,image_h,image_w,c_dim),dtype=tf.float32, name='target'),
  #     tf.TensorSpec(shape=(None,past_TS,a_dim),dtype=tf.float32, name='at_in_seq'),
  #     tf.TensorSpec(shape=(None,future_TS,a_dim),dtype=tf.float32, name='at_target')])
   
  #  dataset=tf.data.Dataset.from_generator(gen_preprocessed_vid, args=[output,past_TS],output_signature=
  #    [(None,past_TS,image_h,image_w,c_dim),
  #     (None,1,image_h,image_w,c_dim),
  #     (None,past_TS-1,image_h,image_w,c_dim),
  #     (None,past_TS,image_h//8,image_w//8,a_dim),
  #     (None,future_TS,image_h,image_w,c_dim),
  #     (None,past_TS,a_dim),
  #     (None,future_TS,a_dim)])
  #  dataset=tf.data.Dataset.from_generator(gen_preprocessed_vid,
  #                                         output_signature=signature_dict)
   return dataset
   

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


# def save_images(images, size, image_path):
#   return imsave((images)*255., size, image_path)
def save_images(images, size, image_path):
    return imsave(np.int32((inverse_transform(images))*255), size, image_path)
    # return imsave(np.int32((images)*255), size, image_path) 

def save_flow_frames(array, size, flow_path):
  opt_flow=np.zeros([size[0]*size[1], array.shape[1],array.shape[2],3])
  opt_flow[..., 0] = array[...,0] *180                       #normalizing by 2*pi 
  opt_flow[..., 1] = 1
  opt_flow[..., 2] = cv2.normalize(array[...,1], None, 0, 255, cv2.NORM_MINMAX)
  # rgb = cv2.cvtColor(opt_flow, cv2.COLOR_HSV2BGR)
  return flowsave(opt_flow, size, flow_path)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  # print("inside the merge function in utils h {}".format(h))
  # print("inside the merge function in utils size {}".format(size))
  img = np.zeros((h * size[0], w * size[1], images.shape[3]),dtype=np.float32)
  # print(h)

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    # print("inside the merge function in utils the shape of image {}".format(image.shape))
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  # print("inside the merge function in utils the shape of img {}".format(img.shape)) 

  return Image.fromarray((np.squeeze(img)).astype(np.uint8))
  # return img

def flowsave(images, size, path):
  return imageio.imwrite(path, cv2.cvtColor(np.float32(merge(images, size)),cv2.COLOR_HSV2BGR))

def imsave(images, size, path):
  return imageio.imwrite(path, merge(images, size))



def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n):
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:,:2,:,0]  = img[:,:2,:,2] = 0
    img[:,:,:2,0]  = img[:,:,:2,2] = 0
    img[:,-2:,:,0] = img[:,-2:,:,2] = 0
    img[:,:,-2:,0] = img[:,:,-2:,2] = 0
    img[:,:2,:,1]  = 255
    img[:,:,:2,1]  = 255
    img[:,-2:,:,1] = 255
    img[:,:,-2:,1] = 255
  else:
    img[:,:2,:,2]  = img[:,:2,:,1] = 0
    img[:,:,:2,2]  = img[:,:,:2,1] = 0
    img[:,-2:,:,2] = img[:,-2:,:,1] = 0
    img[:,:,-2:,2] = img[:,:,-2:,1] = 0
    img[:,:2,:,0]  = 255
    img[:,:,:2,0]  = 255
    img[:,-2:,:,0] = 255
    img[:,:,-2:,0] = 255

  return img

def load_roam_data(vid_dir, length, resize_shape, K, T):
    V_MAX=0.1
    Turn_MAX=1.8
    vid_frames = []
    actions=[]
    low=length[0]
    high=length[1]-(K+T+10)+1
    vid_dir=vid_dir.decode()
    data_dir=vid_dir.split('#')[0]
    zed_dir=os.path.split(data_dir)[0]
    zed_folder=os.path.basename(zed_dir)
    location_dir=os.path.split(zed_dir)[0]
    rosbag_path=os.path.join(location_dir,"Rosbag_"+zed_folder.split('_')[1]+"/")
    # print(zed_dir)
    # print(rosbag_path)
    with open(rosbag_path+"control_actions.txt") as f:
        action_list = f.readlines()
    # print(high)
    # return
    # low = 0
    # high = length - K - T + 1
    assert low <= high, 'video length shorter than K+T'
    stidx = np.random.randint(low, high)
    # stidx_action=stidx//3
    for t in range(1, K+T+1):
        fname =  "{}/left{:08d}.png".format(vid_dir.split('#')[0], t+stidx)
        # print((stidx+t-1)//3)
        action_str= action_list[(stidx+t-1)//3].strip('\n')
        ac=action_str.strip('][').split(', ')
        ac_arr=[float(ac[0]), float(ac[1])]
        # print(ac_arr)
        action=np.multiply(ac_arr,[1/V_MAX,1/Turn_MAX])  ##normalizing control actions.
        im = imageio.imread(fname)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        im=Image.fromarray(im).resize((resize_shape[1], resize_shape[0]))
        tensor=np.ones([resize_shape[0], resize_shape[1],2])
        action_tensor=np.multiply(tensor,action)
        
        im= np.expand_dims(im, axis=0)
        action_tensor= np.expand_dims(action_tensor, axis=0)
        # im=im[...,:]
        # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
        vid_frames.append(im/255.)
        actions.append(action_tensor)
    vid = np.concatenate(vid_frames, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    # print(vid.shape)
    # jerk = accel[1:, ...] - accel[:-1, ...]
    # return vid, diff, accel, actions
    return vid


def load_roam_data_mod(vid_dir, length, resize_shape, K, T):
    V_MAX=0.1
    Turn_MAX=1.8
    vid_frames = []
    actions=[]
    low=length[0]
    high=length[1]-(K+T+10)+1
    vid_dir=vid_dir.decode()
    data_dir=vid_dir.split('#')[0]
    zed_dir=os.path.split(data_dir)[0]
    zed_folder=os.path.basename(zed_dir)
    location_dir=os.path.split(zed_dir)[0]
    rosbag_path=os.path.join(location_dir,"Rosbag_"+zed_folder.split('_')[1]+"/")
    # print(zed_dir)
    # print(rosbag_path)
    with open(rosbag_path+"control_actions.txt") as f:
        action_list = f.readlines()
    # print(high)
    # return
    # low = 0
    # high = length - K - T + 1
    assert low <= high, 'video length shorter than K+T'
    stidx = np.random.randint(low, high)
    # stidx_action=stidx//3
    for t in range(1, K+T+1):
        fname =  "{}/left{:08d}.png".format(vid_dir.split('#')[0], t+stidx)
        # print((stidx+t-1)//3)
        action_str= action_list[(stidx+t-1)//3].strip('\n')
        ac=action_str.strip('][').split(', ')
        ac_arr=[float(ac[0]), float(ac[1])]
        # print(ac_arr)
        action=np.multiply(ac_arr,[1/V_MAX,1/Turn_MAX])  ##normalizing control actions.
        im = imageio.imread(fname)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        im=Image.fromarray(im).resize((resize_shape[1], resize_shape[0]))
        tensor=np.ones([resize_shape[0]//8, resize_shape[1]//8,2])
        action_tensor=np.multiply(tensor,action)
        
        im= np.expand_dims(im, axis=0)
        action_tensor= np.expand_dims(action_tensor, axis=0)
        # im=im[...,:]
        # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
        vid_frames.append(im/255.)
        actions.append(action_tensor)
    vid = np.concatenate(vid_frames, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    # print(actions.shape)
    # jerk = accel[1:, ...] - accel[:-1, ...]
    # output={'vid': vid, 'diff': diff, 'accel': accel, 'actions': actions}
    return vid, diff, accel, actions


def load_roam_data_mod_half_fps(vid_dir, length, resize_shape, K, T):
    V_MAX=0.1
    Turn_MAX=1.8
    vid_frames = []
    actions=[]
    low=length[0]
    high=length[1]-(2*(K+T)+5)+1
    vid_dir=vid_dir.decode()
    data_dir=vid_dir.split('#')[0]
    zed_dir=os.path.split(data_dir)[0]
    zed_folder=os.path.basename(zed_dir)
    location_dir=os.path.split(zed_dir)[0]
    rosbag_path=os.path.join(location_dir,"Rosbag_"+zed_folder.split('_')[1]+"/")
    # print(zed_dir)
    # print(rosbag_path)
    with open(rosbag_path+"control_actions.txt") as f:
        action_list = f.readlines()
    # print(high)
    # return
    # low = 0
    # high = length - K - T + 1
    assert low <= high, 'video length shorter than K+T'
    stidx = np.random.randint(low, high)
    # stidx_action=stidx//3
    for t in range(1, 2*(K+T)+1,2):
        fname =  "{}/left{:08d}.png".format(vid_dir.split('#')[0], t+stidx)
        # print((stidx+t-1)//3)
        action_str= action_list[(stidx+t-1)//3].strip('\n')
        ac=action_str.strip('][').split(', ')
        ac_arr=[float(ac[0]), float(ac[1])]
        # print(ac_arr)
        action=np.multiply(ac_arr,[1/V_MAX,1/Turn_MAX])  ##normalizing control actions.
        im = imageio.imread(fname)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        im=Image.fromarray(im).resize((resize_shape[1], resize_shape[0]))
        tensor=np.ones([resize_shape[0]//8, resize_shape[1]//8,2])
        action_tensor=np.multiply(tensor,action)
        
        im= np.expand_dims(im, axis=0)
        action_tensor= np.expand_dims(action_tensor, axis=0)
        # im=im[...,:]
        # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
        vid_frames.append(im/255.)
        actions.append(action_tensor)
    vid = np.concatenate(vid_frames, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    # print(actions.shape)
    # jerk = accel[1:, ...] - accel[:-1, ...]
    # output={'vid': vid, 'diff': diff, 'accel': accel, 'actions': actions}
    return vid, diff, accel, actions



def optFl2(xt, time_step=10, image_size=[64,64],c_dim=3):
  image_size=[time_step, image_size[0], image_size[1], c_dim]
  mask = np.zeros_like(xt)
  mask[...,1]=255
  mask[...,0]= xt[...,0]*180
  mask[...,2]=cv2.normalize(xt[...,1], None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  
  return rgb