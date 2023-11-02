import tensorflow as tf


def get_dataset(datasize=10000):
    import numpy as np
    return np.random.uniform(0, 1, [datasize, 784]), np.eye(10)[np.random.randint(0, 10, datasize, dtype=np.int)]


def input_fn(mode, epochs, batch_size=256):
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.Dataset.from_tensor_slices(get_dataset(10000)).repeat(epochs)
    elif mode == tf.estimator.ModeKeys.EVAL:
        dataset = tf.data.Dataset.from_tensor_slices(get_dataset(1000))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(get_dataset(100))
    return dataset.batch(batch_size)


def conv_block(input, filters):
    x = tf.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2))(input)
    x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = tf.nn.relu(x)
    return x


def model_fn(features, labels, mode):
    x = tf.keras.layers.Reshape(target_shape=[28, 28, 1])(features)
    x = conv_block(x, filters=32)
    x = conv_block(x, filters=64)
    x = tf.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(x)
    x = tf.layers.Flatten()(x)
    logits = tf.layers.Dense(units=10)(x)

    accuracy, _ = tf.metrics.accuracy(labels=labels, predictions=tf.nn.softmax(logits))
    predictions = {'logits': logits, 'accuracy': accuracy}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
        loss,
        global_step=tf.train.get_or_create_global_step()
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, train_op=train_op, loss=loss)


def main(argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir="./",
            tf_random_seed=1234,
            save_summary_steps=100,
            log_step_count_steps=10
        )
    )
    from greenness_track_toolkit.agent.tensorflow_estimator_hook import AgentSessionHook
    result = estimator.train(
        input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size),
        hooks=[
            AgentSessionHook(
                estimator=estimator,
                username="tangchengfu",
                batch_size=FLAGS.batch_size,
                log_path="../logs",
                server="localhost:16886",
            )
        ]
    )
    print(result)


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer("batch_size", default=128, help="batch_size default 128")
    tf.flags.DEFINE_integer("epochs", default=10, help="epochs default 10")
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()