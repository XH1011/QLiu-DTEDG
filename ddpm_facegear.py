# self package
from Utils.BuildModel2 import *
from Utils.network_utils import *
from Utils.Unet_utils import *
from Utils.utils import *

import pickle as pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 自定义训练步骤
def train_on_step(image_batch):
    image_re = tf.reshape(image_batch, shape=[-1, img_size, 1])
    diffusion_loss = train_loss(image_re)
    return diffusion_loss

def train_loss(image_batch):
    t = tf.random.uniform(
        minval=0, maxval=timesteps,
        shape=(image_batch.shape[0],),
        dtype=tf.int64
    )
    with tf.GradientTape() as tape:
        noise = tf.random.normal(shape=tf.shape(image_batch), dtype=image_batch.dtype)
        image_noise = gdf_util.q_sample(image_batch, t, noise)
        pred_noise = ddpm([image_noise, t], training=True)
        diffusion_loss = mseloss(noise, pred_noise)
    gradients = tape.gradient(diffusion_loss, ddpm.trainable_weights)
    opt.apply_gradients(zip(gradients, ddpm.trainable_weights))
    return diffusion_loss.numpy()

# 反向生成函数
def generate_images(num_images=679):
    samples = tf.random.normal(
        shape=(num_images, img_size, img_channels),
        dtype=tf.float32
    )
    for t in reversed(range(0, timesteps)):
        tt = tf.cast(tf.fill([num_images], t), tf.int64)
        pred_noise = ddpm.predict([samples, tt], verbose=0, batch_size=num_images)
        samples = gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
    return samples

# ---------- 参数设置 ----------
batch_size = 32
num_epochs = 1000
timesteps = 1000
norm_groups = 8
learning_rate = 2e-4
img_size = 128
img_channels = 1
first_conv_channels = 16
channel_multiplier = [4, 2, 1, 0.5]
widths = [first_conv_channels * m for m in channel_multiplier]
has_attention = [False, False, False, False]
num_res_blocks = 2

condition='S4'
gear_type = 'A-S4'
imbalanced_sample='20'
# ---------- 数据加载与处理 ----------
path = '../Data/features/Cross/Cross_'+condition+'/cross_'+gear_type+'-L5-01_train.pkl'
with open(path, 'rb') as f:
    x_train = pickle.load(f)

# 合并并转换为 float32
data_train = np.vstack(x_train).astype(np.float32)
train_ds = tf.data.Dataset.from_tensor_slices(data_train).shuffle(50000).batch(batch_size)

# # ---------- 构建模型 ----------
image_input = layers.Input(shape=(img_size, img_channels), name="generator_input")
time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
ddpm_x = build_model(
    input=image_input,
    time_input=time_input,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels=first_conv_channels,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish)
ddpm = keras.Model([image_input, time_input], ddpm_x)

gdf_util = GaussianDiffusion(timesteps=timesteps)
opt = keras.optimizers.Adam(learning_rate=learning_rate)
mseloss = keras.losses.MeanSquaredError()

print('Network Summary-->')
ddpm.summary()

# ---------- 训练 ----------
loss_list = []
for epoch in range(num_epochs):
    for images_batch in train_ds:
        loss = train_on_step(images_batch)
    loss_list.append(loss)

    if epoch % 100 == 0:
        save_dir = '../Results/DDPM_model/Imbalance_'+imbalanced_sample+'/'+gear_type+'/'
        os.makedirs(save_dir, exist_ok=True)
        ddpm.save_weights(f'{save_dir}model_{epoch}.ckpt')
    print(f'epoch {epoch}, diffusion loss: {loss:.6f}')

# 最后一次保存
save_dir = '../Results/DDPM_model/Imbalance_'+imbalanced_sample+'/'+gear_type+'/'
ddpm.save_weights(f'{save_dir}model_last_{epoch}.ckpt')
np.savetxt('../Results/ddpm_loss_'+gear_type+'.txt',np.array(loss_list))
#
# ---------- 加载 & 生成 ----------
ckpt = f'{save_dir}model_last_999.ckpt'
print('Load weights from', ckpt)
ddpm.load_weights(ckpt)
print('Weights loaded.')

print('Start generate...')
generated_samples = np.squeeze(generate_images(num_images=679))
# generated_samples_test = np.squeeze(generate_images(num_images=291))

# 保存生成结果
out_path = '../Data/features/Imbalance_'+imbalanced_sample+'/Gen_face_gear_'+gear_type+'.pkl'
with open(out_path, 'wb') as f:
    pickle.dump([generated_samples], f, pickle.HIGHEST_PROTOCOL)
print('Generation complete, saved to', out_path)


