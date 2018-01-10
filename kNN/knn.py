import numpy as np
import tensorflow as tf
import sklearn.metrics
from scipy import stats
import sys
sys.path.append('../')
from constant import *

def readData():
    train_x = np.load("../output/data/dataset_train.npy")
    train_y = np.int64(np.load("../output/data/target_train.npy"))
    test_x = np.load("../output/data/dataset_test.npy")
    test_y = np.int64(np.load("../output/data/target_test.npy"))

    return train_x, train_y, test_x, test_y

def one_hot(Y,length):
    NewY=[]
    for i in range(len(Y)):
        content=[]
        num = int(Y[i])
        for i in range(num):
            content.append(0)
        content.append(1)
        for i in range(num+1,length):
            content.append(0)
        NewY.append(content)
    return np.array(NewY)

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test_old = readData()
    Y_train = one_hot(Y_train, CLASSES)
    Y_test = one_hot(Y_test_old, CLASSES)
    xtr = tf.placeholder("float", [None, 453])
    xte = tf.placeholder("float", [453])
    ytr = tf.placeholder(tf.float32, [None, CLASSES])

    k = 5
    #distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    distance = tf.reduce_sum(tf.square(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

    pred = tf.nn.top_k(-distance, k)
    accuracy = 0.
    pred_class = []
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(len(X_test)):
            nn_index = sess.run(pred, feed_dict={xtr: X_train, ytr: Y_train, xte: X_test[i, :]})
            tmp_pred_class = []

            for j in range(k):
                nn_class = np.argmax(Y_train[nn_index[1][j]])
                tmp_pred_class.append(nn_class)
            prediction = stats.mode(tmp_pred_class)[0][0]
            target = np.argmax(Y_test[i])

            print("Test", i, "Prediction:", prediction, "True Class:", target)
            pred_class.append(prediction)
            if prediction == target:
                accuracy += 1. / len(X_test)

        print("Done!")
        print("Accuracy:", accuracy)
        print('F1 score: %f' % sklearn.metrics.f1_score(Y_test_old, np.array(pred_class), average='weighted'))
