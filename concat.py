import os
import pickle
import numpy as np
import tensorflow as tf

diff_dir = '../Data/features/features_diff'
real_dir = '../Data/features/features_real'
save_dir = '../Data/features/features_con'
os.makedirs(save_dir, exist_ok=True)

X_list = ['A', 'B', 'C', 'D', 'E']
Y_list = ['1', '2', '3', '4']

for X in X_list:
    for Y in Y_list:
        name = f"{X}-S{Y}-L5-01_test"
        diff_path = os.path.join(diff_dir, f"diff_{name}.pkl")
        real_path = os.path.join(real_dir, f"real_{name}.pkl")
        save_path = os.path.join(save_dir, f"con_{name}.pkl")

        if not os.path.exists(diff_path) or not os.path.exists(real_path):
            print(f"⚠️ 缺失文件: {name}")
            continue

        with open(diff_path, 'rb') as f:
            diff_feat = pickle.load(f)
        with open(real_path, 'rb') as f:
            real_feat = pickle.load(f)

        # 使用 tf.concat 融合
        diff_tensor = tf.convert_to_tensor(diff_feat, dtype=tf.float32)
        real_tensor = tf.convert_to_tensor(real_feat, dtype=tf.float32)
        fused_tensor = tf.concat([real_tensor, diff_tensor], axis=1)

        # print(diff_tensor.shape, real_tensor.shape,fused_tensor.shape)

        with open(save_path, 'wb') as f:
            pickle.dump(fused_tensor.numpy(), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ 融合完成: {save_path}")
        print(f"形状：{fused_tensor.shape}")