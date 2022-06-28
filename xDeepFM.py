# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：xDeepFM.py
@IDE     ：PyCharm
@Author  ：YDS
@Date    ：2022/6/10 11:35
"""

import os
import scrapbook as sb
import tensorflow as tf
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.models.xDeepFM import XDeepFMModel
from recommenders.models.deeprec.io.iterator import FFMTextIterator
from recommenders.models.deeprec.deeprec_utils import prepare_hparams

tf.get_logger().setLevel('ERROR')


def small_synthetic_dataset():
    EPOCHS_FOR_SYNTHETIC_RUN = 15
    BATCH_SIZE_SYNTHETIC = 128
    RANDOM_SEED = SEED

    data_path = os.path.join(os.getcwd(), 'xdeepfmresources')
    yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
    train_file = os.path.join(data_path, r'synthetic_part_0')
    valid_file = os.path.join(data_path, r'synthetic_part_1')
    test_file = os.path.join(data_path, r'synthetic_part_2')
    output_file = os.path.join(data_path, r'output_small_synthetic_dataset.txt')

    hparams = prepare_hparams(yaml_file,
                              FEATURE_COUNT=1000,
                              FIELD_COUNT=10,
                              cross_l2=0.0001,
                              embed_l2=0.0001,
                              learning_rate=0.001,
                              epochs=EPOCHS_FOR_SYNTHETIC_RUN,
                              batch_size=BATCH_SIZE_SYNTHETIC)
    # print(hparams)
    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator, seed=RANDOM_SEED)
    #  模型训练前表现
    print(model.run_eval(test_file))

    history = model.fit(train_file, valid_file)
    print(history)

    #  模型训练后表现
    res_syn = model.run_eval(test_file)
    print(res_syn)

    #  模型测试集表现
    pre = model.predict(test_file, output_file)
    print(pre)

    print(history)


def small_criteo_dataset():
    EPOCHS_FOR_CRITEO_RUN = 30
    BATCH_SIZE_CRITEO = 4096
    RANDOM_SEED = SEED
    data_path = os.path.join(os.getcwd(), 'xdeepfmresources')
    yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
    train_file = os.path.join(data_path, r'cretio_tiny_train')
    valid_file = os.path.join(data_path, r'cretio_tiny_valid')
    test_file = os.path.join(data_path, r'cretio_tiny_test')
    output_file = os.path.join(data_path, r'output_small_criteo_dataset.txt')

    hparams = prepare_hparams(yaml_file,
                              FEATURE_COUNT=2300000,
                              FIELD_COUNT=39,
                              cross_l2=0.01,
                              embed_l2=0.01,
                              layer_l2=0.01,
                              learning_rate=0.002,
                              batch_size=BATCH_SIZE_CRITEO,
                              epochs=EPOCHS_FOR_CRITEO_RUN,
                              cross_layer_sizes=[20, 10],
                              init_value=0.1,
                              layer_sizes=[20, 20],
                              use_Linear_part=True,
                              use_CIN_part=True,
                              use_DNN_part=True)

    model = XDeepFMModel(hparams, FFMTextIterator, seed=RANDOM_SEED)

    #  模型训练前表现
    print(model.run_eval(test_file))

    his = model.fit(train_file, valid_file)
    print(his)

    #  模型训练后表现
    res_real = model.run_eval(test_file)
    print(res_real)
    sb.glue("res_real", res_real)

    #  模型测试集表现
    pre = model.predict(test_file, output_file)
    print(pre)


if __name__ == '__main__':
    # 微软合成数据集
    small_synthetic_dataset()
    # Criteo数据集
    small_criteo_dataset()
