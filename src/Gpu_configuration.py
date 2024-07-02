import tensorflow as tf
#with tf.device('/GPU:6'):
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        #tf.config.experimental.set_memory_growth(gpu,True) #Second option for dynamically setting memroy growth 