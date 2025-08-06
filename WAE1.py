import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle
from swd_util import sw_loss
def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)



def train_on_step(images_batch):
    image_batch = tf.reshape(images_batch, shape=[-1, 1024, 1])
    loss=train_loss(image_batch)

    return loss

def train_loss(image_batch):
    with tf.GradientTape() as tape:
        # model
        recon_image = dcae(image_batch, training=True)
        #loss in sliced wasserstein
        image_batch,recon_image=tf.squeeze(image_batch),tf.squeeze(recon_image)
        recon_loss = sw_loss(true_distribution=image_batch,
                             generated_distribution=recon_image,
                             num_projections=image_batch.shape[0],
                             batch_size=image_batch.shape[0])
        # diff = recon_image - image_batch
        # recon_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        # print('loss:',image_batch.shape,images_batch.shape,recon_image.shape)
        total_loss = recon_loss
    gradients = tape.gradient(total_loss, dcae.trainable_weights)
    opt.apply_gradients(zip(gradients, dcae.trainable_weights))

    return total_loss.numpy()

#查看时间
import time
start_time = time.time()

# batch_size=32 #batch_size=32 (大样本)
batch_size=2 #batch_size=2 （小样本）
num_epochs=10000
learning_rate=2e-4 #learning_rate=2e-4
img_size=1024
img_channels=1
endim=128
interdim=256

# Global Settings
fault_name='C2_4_Tt'
file_name='./Data/Splite/'+fault_name+'.pkl'
with open(file_name, 'rb') as f:
    data_train,data_test = pickle.load(f)

    np.random.seed(23)
    # 随机选择 k个样本
    mix = [i for i in range(len(data_train))]
    np.random.shuffle(mix)
    train_x = data_train[mix]
    random_x = train_x[0:10, ]
data_train=tf.cast(random_x,dtype=tf.float32)
# data_train=tf.cast(random_x,dtype=tf.float32)
train_ds=tf.data.Dataset.from_tensor_slices(data_train).shuffle(10000).batch(batch_size)

#inputs
inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
# 2，神经网络
layers=tf.keras.layers
# ### Encoder
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(inputs_)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x) #-1,16,64

x = layers.Flatten()(x) #1024
x=layers.Dense(units=interdim, activation=leaky_relu)(x)
enc=layers.Dense(units=1024, activation=leaky_relu)(x) #256

x = tf.reshape(enc, [-1, 16, 64])
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)
x = layers.UpSampling1D(2)(x)
rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=leaky_relu)(x)

# # # #Build model
dcae=keras.Model(inputs_, rx)
#
# # # # Opimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
print('Network Summary-->')
dcae.summary()


# --------------->>>Training Phase<<<---------------------------
# Run
loss_list=[]
# total_batch=350
total_batch = int(len(data_train) / batch_size)
for epoch in range(num_epochs):
    ave_cost=0
    for images_batch in train_ds:
        # print(images_batch.shape)
        loss=train_on_step(images_batch)
        ave_cost+=loss/total_batch
    print("Epoch:", (epoch + 1), "cost =", ave_cost)
    loss_list.append(ave_cost)

    # Save the model weights every 100 epoches
    if epoch%10==0:
      save_dir='./Results/wae_model/model'+fault_name+'/'
      os.makedirs(save_dir, exist_ok=True)
      dcae.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')

    # save loss
    loss_curve = np.array(loss_list)
    save_loss = './Results/wae_model/loss/'
    os.makedirs(save_loss, exist_ok=True)
    np.savetxt(save_loss + 'wae_loss'+fault_name+'.txt', loss_curve)

# Save the model weights in the last step
save_dir='./Results/wae_model/model'+fault_name+'/'
dcae.save_weights(save_dir+'model_last_'+str(epoch)+'.ckpt')
print('Optimization Finished')
end_time = time.time()
print(f"代码运行时间: {end_time - start_time:.6f} 秒")

