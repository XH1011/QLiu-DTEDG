import os
import pandas as pd

import aug
import data_utils
import load_methods


def get_files(root, dataset, faults, fault_label, signal_size, condition=3):
    data, actual_labels = [], []
    data_load = getattr(load_methods, dataset)
    
    for _, name in enumerate(faults):
        data_dir = os.path.join(root, 'condition_%d' % condition, name)

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)

            start, end = 0, signal_size
            while end <= signal.shape[0]:
                data.append(signal[start:end])
                actual_labels.append(fault_label[name])
                start += signal_size
                end += signal_size

    return data, actual_labels


# def get_files_processed(root, dataset, faults, fault_label, signal_size, condition=3):
#     data, actual_labels = [], []
#     data_load = getattr(load_methods, dataset)
#
#
#     for _, name in enumerate(faults):
#         data_dir = os.path.join(root, 'condition_%d' % condition, name)
#
#         if not os.path.exists(data_dir):
#             print(f"Warning: Fault directory not found: {data_dir}")
#             continue
#
#         for item in os.listdir(data_dir):
#             if item.startswith('.'):  # Skip hidden files like .DS_Store
#                 continue
#             item_path = os.path.join(data_dir, item)
#             signal = data_load(item_path)
#
#             if signal_size is None:
#                 data.append(signal)
#                 actual_labels.append(fault_label[name])
#
#     return data, actual_labels


def get_files_processed(root, dataset, faults, fault_label, signal_size, condition=3):
    data, actual_labels = [], []
    data_load = getattr(load_methods, dataset)

    for name in faults:
        fault_dir = os.path.join(root, f'condition_{condition}', name)
        if not os.path.exists(fault_dir):
            continue

        for file in os.listdir(fault_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(fault_dir, file)
                signal = data_load(file_path)  # signal.shape 应为 (970, 128)
                if signal is None:
                    continue

                # 按行拆分样本
                for i in range(signal.shape[0]):
                    data.append(signal[i])  # 取第 i 个样本（形状为 (128,)）
                    actual_labels.append(fault_label[name])

    return data, actual_labels

def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()

        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms


class dataset(object):
    
    def __init__(self, args, dataset, source_idx, condition=2, balance_data=False, test_size=0.2):

        # 其他初始化...
        self.args = args
        data_root = os.path.join(args.data_dir, dataset)
        faults = args.faults[source_idx]
        signal_size = args.signal_size
        normlize_type = args.normlize_type
        fault_label = args.fault_label
        self.label_set = args.label_sets[source_idx]
        self.random_state = args.random_state
        self.balance_data = balance_data
        self.test_size = test_size

        # 检查路径是否存在
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data directory not found: {data_root}")

        # Step 1: 加载原始数据
        self.data, self.actual_labels = get_files_processed(data_root, dataset, faults, fault_label, signal_size, condition=condition)

        # 检查数据是否为空
        if len(self.data) == 0:
            raise ValueError(f"No data loaded for dataset {dataset}, condition {condition}, faults {faults}")

        # 检查样本数量和形状
        print(f"Total samples loaded: {len(self.data)}")
        if len(self.data) > 0:
            print(f"Sample shape: {self.data[0].shape}")  # 应为 (128,)

        # Step 2: 定义数据变换（预处理）
        self.transform = data_transforms(normlize_type)



    def data_preprare(self, is_src=False):
        data_pd = pd.DataFrame({"data": self.data, "actual_labels": self.actual_labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, transform=self.transform['train'])
            return train_dataset
        else:
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size, label_set=self.label_set)
            train_dataset = data_utils.dataset(list_data=train_pd, transform=self.transform['train'])
            val_dataset = data_utils.dataset(list_data=val_pd, transform=self.transform['val'])
            return train_dataset, val_dataset
