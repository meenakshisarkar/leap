import numpy as np
import tensorflow as tf
from math import ceil
import tensorflow_datasets as tfds
np.random.seed(77)
tf.random.set_seed(77)
AUTOTUNE = tf.data.AUTOTUNE
def normalize_image(image):
    """ normalize image to [-1, 1] """
    image = tf.cast(image, dtype=tf.float32)
    return (image /127.5 - 1.0)
    # """ normalize image to [0, 1] """
    # return (image /255)  
def normalize_image_v2(image):
    """ normalize image to [-1, 1] """
    image = tf.cast(image, dtype=tf.float32)
    # return (image /127.5 - 1.0)
    # """ normalize image to [0, 1] """
    return (image /255)  


def load_train_data(strategy,dataname='bair_robot_pushing_small',batch=16):
  if dataname== 'bair_robot_pushing_small':
    ds, metadata=tfds.load(dataname, split='train',with_info=True,)
    print(metadata)
    get_size = lambda name: metadata.splits.get(name).num_examples
    num_train_samples=get_size('train')
  elif dataname== 'kth':
    ds, metadata=tfds.load(dataname, split='train',with_info=True,)
    print(metadata)
    get_size = lambda name: metadata.splits.get(name).num_examples
    num_train_samples=get_size('train')
  train_step= ceil(num_train_samples / batch)
  # ds=list(ds.take(num_train_samples))
  # images=ds['image_main']
  # train_images=images.map(normalize_image,num_parallel_calls=AUTOTUNE)
  train_dframes=ds.cache()
  train_dframes=train_dframes.shuffle(buffer_size=512)
  train_dframes=train_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
  
  
  train_ds = strategy.experimental_distribute_dataset(train_dframes)
  return train_ds, train_step
def load_train_data_w_Val(strategy,dataname='bair_robot_pushing_small',batch=16):
  if dataname== 'bair_robot_pushing_small':
    ds, metadata=tfds.load(dataname, split='train[:-100]',data_dir= '../../../../tensorflow_datasets/',download=False, with_info=True,)
    print(metadata)
    val_ds =tfds.load(dataname, split='train[-100:]',data_dir= '../../../../tensorflow_datasets/',download=False)
    get_size = lambda name: metadata.splits.get(name).num_examples

    num_train_samples=get_size('train')
    train_step= ceil(num_train_samples / batch)
    # ds=list(ds.take(num_train_samples))
    # images=ds['image_main']
    # train_images=images.map(normalize_image,num_parallel_calls=AUTOTUNE)
    train_dframes=ds.cache()
    train_dframes=train_dframes.shuffle(buffer_size=512)
    train_dframes=train_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
    val_dframes=val_ds.cache()
    val_dframes=val_dframes.shuffle(buffer_size=100)
    val_dframes=val_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
    train_ds = strategy.experimental_distribute_dataset(train_dframes)
    validation_ds=strategy.experimental_distribute_dataset(val_dframes)
  # elif dataname== 'kth':
  #   import data.kth_dataset as KTH
  #   train_ds=KTH.kth(strategy,batch,1)
  #   validation_ds=KTH.kth(strategy,batch,1)
  #   train_step=None
    
  elif dataname== 'roam':
    ds, metadata=tfds.load(dataname, split='train[:-100]',data_dir= '../../../tensorflow_datasets/',download=False, with_info=True)#,shuffle_files=True)
    print(metadata)
    val_ds =tfds.load(dataname, split='train[-100:]',data_dir= '../../../tensorflow_datasets/',download=False)#,shuffle_files=True)
    get_size = lambda name: metadata.splits.get(name).num_examples
    num_train_samples=get_size('train')
    train_step= ceil(num_train_samples / batch)
    # ds=list(ds.take(num_train_samples))
    # images=ds['image_main']
    # train_images=images.map(normalize_image,num_parallel_calls=AUTOTUNE)
    train_dframes=ds.cache()
    # train_dframes=ds.cache()
    train_dframes=train_dframes.shuffle(buffer_size=5120,reshuffle_each_iteration=True).repeat()
    train_dframes=train_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
    val_dframes=val_ds.cache().repeat()
    val_dframes=val_dframes.shuffle(buffer_size=100)
    val_dframes=val_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
  
  
  train_ds = strategy.experimental_distribute_dataset(train_dframes)
  validation_ds=strategy.experimental_distribute_dataset(val_dframes)

  return train_ds, train_step, validation_ds


def load_test_data(strategy,dataname='bair_robot_pushing_small',batch=1):
  if dataname== 'bair_robot_pushing_small':
    ds, metadata=tfds.load(dataname, split='test',with_info=True,)
  elif dataname== 'kth':
    ds, metadata=tfds.load(dataname, split='test',with_info=True,)
  elif dataname== 'roam_test':
    ds, metadata=tfds.load(dataname, split='test',with_info=True,)
  print(metadata)
  get_size = lambda name: metadata.splits.get(name).num_examples
  num_test_samples=get_size('test')
  test_step= ceil(num_test_samples / batch)
  # ds=list(ds.take(num_test_samples))
  # images=ds['image_main']
  # test_images=images.map(normalize_image,num_parallel_calls=AUTOTUNE)
  test_dframes=ds.cache()
  # test_dframes=test_dframes.shuffle(buffer_size=512)
  test_dframes=test_dframes.batch(batch,drop_remainder=True).prefetch(AUTOTUNE)
    
    
  test_ds = strategy.experimental_distribute_dataset(test_dframes)
  return test_ds, test_step
  

