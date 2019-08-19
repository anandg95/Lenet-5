# This is where we will be training and evaluating the model

import tensorflow as tf
import preprocess_data
from model import LeNet
from sklearn.utils import shuffle

X_train, y_train, X_validation, y_validation, X_test, y_test = (
    preprocess_data.preprocess_data()
)

# HYPERSSS!
EPOCHS = 10
BATCH_SIZE = 128
lr = 0.001

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(tf.int32, shape=(None))
one_hot_y = tf.one_hot(y, 10)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.math.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

correct_pred = tf.math.equal(tf.math.argmax(logits, 1), tf.argmax(one_hot_y, 1))  # bool
accuracy_op = tf.reduce_mean(
    tf.cast(correct_pred, tf.float32)
)  # put all trues to 1, falses to 0, and find mean
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    total_correct = 0

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = (
            X_data[offset : offset + BATCH_SIZE],
            y_data[offset : offset + BATCH_SIZE],
        )
        correct = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        total_correct += correct * BATCH_SIZE

    return total_correct / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = (
                X_train[offset : offset + BATCH_SIZE],
                y_train[offset : offset + BATCH_SIZE],
            )
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print(f"Epoch : {i+1} Accuracy : {validation_accuracy}")

    saver.save(sess, "LeNet-5")
    print("Model saved.")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("."))
    test_accuracy = evaluate(X_test, y_test)
    print(f"Test accuracy attained : {test_accuracy}")
