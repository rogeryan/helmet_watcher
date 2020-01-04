import tensorflow as tf
from core import ssd
from parse_pascal_voc import ParsePascalVOC
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, \
    CHANNELS, EPOCHS, cls_loss_weight, reg_loss_weight, save_model_dir, NUM_CLASSES
from core.label_anchors import LabelAnchors
from core.loss import SmoothL1Loss
import math
import os
import logging
import time


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get datasets
    t0 = time.time()
    parse = ParsePascalVOC()
    train_dataset, test_dataset, train_count, test_count = parse.split_dataset()
    t1 = time.time()
    logger.info(f'数据加载完成，耗时：{t1 - t0}')
    # initialize model
    model = ssd.SSD()
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    t0 = time.time()
    logger.info(f'模型配置完成，耗时：{t0- t1}')
    model.summary()

    # metrics
    train_class_metric = tf.keras.metrics.Accuracy()
    train_box_metric = tf.keras.metrics.MeanAbsoluteError()
    test_class_metric = tf.keras.metrics.Accuracy()
    test_box_metric = tf.keras.metrics.MeanAbsoluteError()

    # optimizer
    optimizer = tf.keras.optimizers.Adadelta()

    # loss
    train_cls_loss = tf.keras.losses.CategoricalCrossentropy()
    # train_reg_loss = SmoothL1Loss()
    calculate_train_loss = tf.keras.metrics.Mean()
    test_cls_loss = tf.keras.losses.CategoricalCrossentropy()
    # test_reg_loss = SmoothL1Loss()
    calculate_test_loss = tf.keras.metrics.Mean()

    # @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            anchors, class_preds, box_preds = model(images)
            label_anchors = LabelAnchors(anchors=anchors, labels=labels, class_preds=class_preds)
            box_target, box_mask, cls_target = label_anchors.get_results()
            logger.info(
                f'class_preds type:{type(cls_target)}, shape: {class_preds.shape}')
            logger.info(f'cls_target type:{type(cls_target)}, shape: {cls_target.shape}')
            cls_target = tf.dtypes.cast(cls_target, tf.int32)
            cls_target_onehot = tf.one_hot(indices=cls_target, depth=NUM_CLASSES + 1)
            cls_loss = train_cls_loss(y_pred=class_preds, y_true=cls_target_onehot)
            train_reg_loss = SmoothL1Loss(mask=box_mask)
            reg_loss = train_reg_loss(box_target, box_preds)
            loss = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        calculate_train_loss(loss)
        class_preds_argmax = tf.math.argmax(class_preds, axis=-1)
        train_class_metric.update_state(y_true=cls_target, y_pred=class_preds_argmax)
        train_box_metric.update_state(y_true=box_target, y_pred=box_preds * box_mask)


    for epoch in range(EPOCHS):
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}, mse: {:.5f}".format(epoch + 1,
                                                                                                  EPOCHS,
                                                                                                  step,
                                                                                                  math.ceil(train_count / BATCH_SIZE),
                                                                                                  calculate_train_loss.result(),
                                                                                                  train_class_metric.result(),
                                                                                                  train_box_metric.result()))


    tf.saved_model.save(model, save_model_dir)
