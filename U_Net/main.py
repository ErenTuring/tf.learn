#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:55:45 2017

@author: leilei
"""

import tensorflow as tf
# import numpy as np
import nets
import batch_Data


def L2_loss(weight_decay_rate):
    weights = [
        var for var in tf.trainable_variables()
        if var.name.endswith('weights:0')
    ]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
    return l2_loss * weight_decay_rate


def Loss(logits, labels):
    ''' None class weight '''
    labels = tf.to_float(labels)
    logits = tf.add(logits, tf.constant(1e-4))
    segment_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return segment_loss


def Loss_func(*losses):
    return tf.reduce_sum(losses)


def Accuracy(pred, labels):
    label = tf.argmax(labels, axis=3)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), dtype=tf.float32))
    return accuracy


def mean_IOU(pred, labels, num_classes):
    labels = tf.argmax(labels, axis=3)
    Confus_mat = tf.metrics.mean_iou(
        labels=labels, predictions=pred,
        num_classes=num_classes)[1]  # [0] => mean_iou

    recall = tf.div(Confus_mat[1, 1],
                    tf.add(Confus_mat[1, 1], Confus_mat[1, 0]))
    precision = tf.div(Confus_mat[1, 1],
                       tf.add(Confus_mat[1, 1], Confus_mat[0, 1]))
    mean_iou = tf.div(
        Confus_mat[1, 1],
        tf.add_n([Confus_mat[1, 1], Confus_mat[1, 0], Confus_mat[0, 1]]))

    return precision, recall, mean_iou


def train(loss, learning_rate, learning_rate_decay_steps,
          learning_rate_decay_rate, global_step):
    ''' globale_step automatic update +1  and update lr '''
    decay_lr = tf.train.exponential_decay(
        learning_rate,
        global_step,
        learning_rate_decay_steps,
        learning_rate_decay_rate,
        staircase=True)
    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(decay_lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def main():
    # dataset_path = r'/home/qiji/Code/Coding/ISPRS_data_3'
    # train_list_path = r'/home/qiji/Code/Coding/ISPRS_data_3/train.txt'
    # valid_list_path = r'/home/qiji/Code/Coding/ISPRS_data_3/valid.txt'
    # model_path = r'/home/qiji/Code/Coding/ISPRS_data_3/model/model.ckpt'
    dataset_path = r'C:/WorkSpace/myTF/ISPRS_data_3'
    train_list_path = r'C:/WorkSpace/myTF/ISPRS_data_3/train.txt'
    valid_list_path = r'C:/WorkSpace/myTF/ISPRS_data_3/valid.txt'
    model_path = r'C:/WorkSpace/myTF/ISPRS_data_3/model/model.ckpt'

    batch_size = 8
    img_size = 256
    max_iter = 1000  # 40001
    learning_rate = 1e-3
    class_number = 3
    ''' batch_data params '''
    dataset_train_param = batch_Data.Data(dataset_path, train_list_path,
                                          class_number)
    # dataset_mean_vector = dataset_train_param.meanvector
    dataset_valid_param = batch_Data.Data(dataset_path, valid_list_path,
                                          class_number)

    image = tf.placeholder(
        tf.float32,
        shape=[batch_size, img_size, img_size, 3],
        name='image')
    label = tf.placeholder(
        tf.int32,
        shape=[batch_size, img_size, img_size, class_number],
        name='label')
    is_training = tf.placeholder(tf.bool, name='is_training')

    with tf.name_scope('model'):
        fcn1 = nets.Model()
        fcn1.build(image, class_number, is_training)

        # tf.add_to_collection('new_score_1', fcn1.new_score_1)
        tf.add_to_collection('new_score_2', fcn1.new_score_2)
        tf.add_to_collection('pred', fcn1.pred_2)

    with tf.name_scope('loss'):
        loss = Loss_func(
            # Loss(fcn1.new_score_1, label), Loss(fcn1.new_score_2, label),
            Loss(fcn1.new_score_2, label),
            L2_loss(5e-4))

    global_step = tf.Variable(tf.constant(0))
    train_op = train(loss, learning_rate, 10000, 0.1, global_step)

    sess = tf.Session()
    accuracy = Accuracy(pred=fcn1.pred_2, labels=label)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # 合并到Summary中
    merged = tf.summary.merge_all()
    # 选定可视化存储目录
    train_writer = tf.summary.FileWriter(nets.log_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init_op)
    for step in range(1, max_iter):
        #        print(sess.run(global_step))
        train_imgs, train_labs = dataset_train_param.next_batch(
                                batch_size=batch_size, flag='train')
        feed_dict = {image: train_imgs, label: train_labs, is_training: True}

        if step % 10 == 0:
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print('step : %d, loss : %g' % (step, train_loss))
            if step % 50 == 0:
                summary = sess.run(merged, feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                if step % 100 == 0:
                    valid_imgs, valid_labs = dataset_valid_param.next_batch(
                        batch_size=batch_size, flag='valid')
                    feed_dict = {
                        image: valid_imgs,
                        label: valid_labs,
                        is_training: False
                    }
                    accu = round(sess.run(accuracy, feed_dict=feed_dict), 2)
                    print('Accuracy: %.2f' % accu)
                    saver.save(
                        sess,
                        save_path=model_path,
                        global_step=step)

        else:
            sess.run(train_op, feed_dict=feed_dict)

    sess.close()


if __name__ == '__main__':
    main()
