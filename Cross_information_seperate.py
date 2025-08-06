import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import time
# visiualize
def scatter(x, colors, n_class):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_class))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_class):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
def t_sne(x, y, n_class):
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    sns.set_style('whitegrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    from sklearn.manifold import TSNE
    digits_proj = TSNE(random_state=128).fit_transform(x)

    scatter(digits_proj, y, n_class)

    # plt.savefig(savename)
def TFData_preprocessing(x1,x2,batch_size):
    x=tf.data.Dataset.from_tensor_slices((x1,x2))
    x=x.shuffle(231).batch(batch_size)
    return x
#1. 互信息化之前的t-SNE
# for condition in ['S1','S2','S3','S4']:
#     features=[]
#     for fault_type in ['A','B','C','D','E']:
#         # with open('../Data/features/features_con/con_A-S1-L5-01_train.pkl', 'rb') as f:
#         #     features_train = pickle.load(f)
#
#         with open('../Data/features/features_real/real_'+fault_type+'-'+condition+'-L5-01_test.pkl', 'rb') as f:
#             features_test = pickle.load(f)
#             features.extend(features_test)
#             print(fault_type,features_test.shape)
#     features = np.array(features)
#     # y_train = np.array([0]*20)
#     y_test = np.array([0]*291+[1]*291+[2]*291+[3]*291+[4]*291)
#
#     # y=np.hstack((y_train, y_test))
#     # x=np.vstack((features_train, features_test))
#     t_sne(features, y_test, 5)
#     plt.show()

#2. 进行互信息化
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#定义两个 encoder（可共享参数，也可不共享）
def build_encoder(name):
    return tf.keras.Sequential([
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(interdim)  # feature vector
    ], name=name)
#InfoNCE-style Mutual Info Loss
# def infonce_loss(z_sim, z_real, temperature=0.1):
#     # z_sim, z_real ∈ [B, d]
#     z_sim = tf.math.l2_normalize(z_sim, axis=1)
#     z_real = tf.math.l2_normalize(z_real, axis=1)
#
#     # 正例相似度：对角线
#     logits = tf.matmul(z_real, z_sim, transpose_b=True)  # [B, B]
#     labels = tf.range(tf.shape(logits)[0])  # 正确匹配的索引
#
#     # 计算 InfoNCE 损失
#     loss = tf.keras.losses.sparse_categorical_crossentropy(
#         labels, logits / temperature, from_logits=True
#     )
#     return tf.reduce_mean(loss)

def infonce_loss(z_sim, z_real, temperature=0.07):
    z_sim = tf.math.l2_normalize(z_sim, axis=1)
    z_real = tf.math.l2_normalize(z_real, axis=1)
    logits = tf.matmul(z_real, z_sim, transpose_b=True)
    labels = tf.range(tf.shape(logits)[0])
    logits = logits / temperature
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def train_on_step(images_batch1,images_batch2):
    image_batch1 = tf.reshape(images_batch1, shape=[-1, feature_dim1, 1])
    image_batch2 = tf.reshape(images_batch2, shape=[-1, feature_dim2, 1])
    loss=train_loss(image_batch1, image_batch2)

    return loss
def train_loss(X1,X2):
    with tf.GradientTape() as tape:
        z1 = encoder_sim(X1, training=True)
        z2 = encoder_real(X2, training=True)
        mi_loss = infonce_loss(z1, z2)
        # align_loss = tf.reduce_mean(tf.square(z1 - z2))
        total_loss = mi_loss  # 你也可以加上重建 loss / 分类 loss 等
        # total_loss = mi_loss + 0.5 * align_loss

    # 训练示意代码（custom training loop）
    grads = tape.gradient(total_loss, encoder_real.trainable_variables + encoder_sim.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder_real.trainable_variables + encoder_sim.trainable_variables))

    return total_loss.numpy()

start_time = time.time()
learning_rate=1e-4
batch_size = 16
feature_dim1=256
feature_dim2=256
interdim=128
num_epochs=10000
encoder_sim = build_encoder("Encoder_Sim")  # for X1
encoder_real = build_encoder("Encoder_Real")  # for X2
optimizer=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)

condition='S4'
#load data x1 (sim)
features_x1=[]
for fault_type in ['B']:
    with open('../Data/features/features_sim/sim_F-'+fault_type+'-'+condition+'-L5-01_train.pkl', 'rb') as f:
        x1 = pickle.load(f)
        features_x1.extend(x1)
        print(fault_type,x1.shape)
features_x1 = np.array(features_x1)

# laod data x2 (real)
features_x2=[]
for fault_type in ['B']:
    with open('../Data/features/features_real/real_'+fault_type+'-'+condition+'-L5-01_train.pkl', 'rb') as f:
        x2 = pickle.load(f)
        features_x2.extend(x2)
        print(fault_type,x2.shape)
features_x2 = np.array(features_x2)
train_db = TFData_preprocessing(features_x1,features_x2,batch_size)

for images_batch1,images_batch2 in train_db:
    print(images_batch1.shape, images_batch2.shape)

# Run
loss_list=[]
total_batch = int(len(features_x1) / batch_size)
for epoch in range(num_epochs):
    ave_cost=0
    for images_batch1,images_batch2 in train_db:
        loss=train_on_step(images_batch1,images_batch2)
        ave_cost+=loss/total_batch
    print("Epoch:", (epoch + 1), "cost =", ave_cost)
    loss_list.append(ave_cost)

    if epoch % 5000 == 0:
        # 保存 encoder_sim 和 encoder_real 的模型权重
        encoder_sim.save('../Results/CrossInformation_se_S4/encoder_sim_weights_'+str(epoch)+'_'+fault_type)
        encoder_real.save('../Results/CrossInformation_se_S4/encoder_real_weights_'+str(epoch)+'_'+fault_type)
# 训练结束后保存最终模型
encoder_sim.save('../Results/CrossInformation_se_S4/encoder_sim_weights_' + str(epoch+1)+'_'+fault_type)
encoder_real.save('../Results/CrossInformation_se_S4/encoder_real_weights_' + str(epoch+1)+'_'+fault_type)

# plot loss over 1000 epochs
loss=np.array(loss_list)
plt.plot(loss_list)
plt.show()

# 加载整个模型
encoder_real = tf.keras.models.load_model('../Results/CrossInformation_se_S4/encoder_real_weights_10000_'+fault_type)
print('Optimization Finished')
end_time = time.time()
print(f"代码运行时间: {end_time - start_time:.6f} 秒")

# ######testing stage........
CrossFeature_train_real = encoder_real(features_x2, training=False)
print(CrossFeature_train_real.shape)

# laod data x2 (real) test
features_x2_test=[]
for fault_type in ['B']:
    with open('../Data/features/features_real/real_'+fault_type+'-'+condition+'-L5-01_test.pkl', 'rb') as f:
        x2_test = pickle.load(f)
        features_x2_test.extend(x2_test)
features_x2_test = np.array(features_x2_test)
CrossFeature_test_real = encoder_real(features_x2_test, training=False)
print(CrossFeature_test_real.shape)

# 保存特征 训练
with open('../Data/features/Cross/Cross_S4/cross_'+fault_type+'-S4-L5-01_train.pkl', 'wb') as f:
    pickle.dump(CrossFeature_train_real[0:20,:], f, pickle.HIGHEST_PROTOCOL)

# 保存特征 测试
with open('../Data/features/Cross/Cross_S4/cross_'+fault_type+'-S4-L5-01_test.pkl', 'wb') as f:
    pickle.dump(CrossFeature_test_real[0:291*1,:], f, pickle.HIGHEST_PROTOCOL)
print("Feature saved successfully!")