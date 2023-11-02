from tensorflow import keras

from greenness_track_toolkit import Agent


def get_dataset(datasize=10000):
    import numpy as np
    return np.random.uniform(0, 1, [datasize, 784]), np.eye(10)[np.random.randint(0, 10, datasize, dtype=np.int)]


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_dataset(10000), get_dataset(1000)
    model = keras.models.Sequential(layers=[
        keras.layers.Reshape(target_shape=[28, 28, 1]),
        keras.layers.Conv2D(filters=32, kernel_size=3),
        keras.layers.MaxPooling2D(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=64, kernel_size=3),
        keras.layers.MaxPooling2D(),
        keras.layers.ReLU(),
        keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=10, activation=keras.activations.relu)
    ])
    opt = keras.optimizers.SGD()
    model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy())
    model.build(input_shape=[None, 784])
    model.summary()
    with Agent(
            username="tangchengfu",
            model=model,
            batch_size=256,
            log_path="../logs",
    ):
        model.fit(x=x_train, y=y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))