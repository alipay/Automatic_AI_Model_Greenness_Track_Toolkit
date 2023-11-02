import tensorflow as tf
from tensorflow.python.training.training_util import get_or_create_global_step


def conv_block(input, filters):
    x = tf.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2))(input)
    x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = tf.nn.relu(x)
    return x


def build_model():
    input_x = tf.placeholder(dtype="float", shape=[None, 784])
    y = tf.placeholder(dtype="float", shape=[None, 10])
    x = tf.keras.layers.Reshape(target_shape=[28, 28, 1])(input_x)
    x = conv_block(x, filters=32)
    x = conv_block(x, filters=64)
    x = tf.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = tf.layers.Flatten()(x)
    logits = tf.layers.Dense(units=10)(x)
    y_ = tf.nn.softmax(logits=logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    return input_x, y, y_, loss


def get_dataset(datasize=10000):
    import numpy as np
    return np.random.uniform(0, 1, [datasize, 784]), np.eye(10)[np.random.randint(0, 10, datasize, dtype=np.int)]


if __name__ == '__main__':
    from greenness_track_toolkit import Agent
    import numpy as np

    checkpoint_path = "./model/model"
    epochs = 1
    batch_size = 16
    buffer_size = 0
    datasize = 10000
    train_dataset = tf.data.Dataset.from_tensor_slices(
        get_dataset(datasize)
    ).batch(
        batch_size=batch_size).prefetch(buffer_size).repeat()

    train_dataset_iter = train_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        x_, y_, y_pred, loss_ = build_model()
        acc_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_pred, 1)), tf.float32))

        global_step = get_or_create_global_step()
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss_, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        batch_per_epoch = datasize // batch_size
        with Agent(
                # server="localhost:16886",
                username="tangchengfu.tcf",
                session=sess,
                batch_size=batch_size,
                log_path="../logs",
        ) as agent:
            for epoch in range(0, epochs):
                saver.save(sess, save_path=checkpoint_path, global_step=global_step)
                for step in range(0, batch_per_epoch):
                    x, y = sess.run(fetches=[train_dataset_iter])[0]
                    y_p, loss, acc, _ = sess.run(fetches=[y_pred, loss_, acc_, opt], feed_dict={
                        x_: x, y_: y
                    })
                    print(step, loss, acc)