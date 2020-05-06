import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
tfds.disable_progress_bar()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    one_hot_label = [label == 0, label == 1]

    return image, one_hot_label


def learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('model/train_curves.png')
    plt.show()


if __name__ == '__main__':

    save_model = True
    IMG_SIZE = 160  # All images will be resized to 160x160
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    train_split = tfds.core.ReadInstruction('train', to=80, unit='%')
    val_split = tfds.core.ReadInstruction('train', from_=80, to=90, unit='%')
    test_split = tfds.core.ReadInstruction('train', from_=90, unit='%')

    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=[train_split, val_split, test_split],
        with_info=True,
        as_supervised=True,
    )

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(2)

    model = tf.keras.Sequential([
      base_model,
      global_average_layer,
      prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    initial_epochs = 10

    history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

    # Fine tuning the model
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                  metrics=['accuracy'])

    model.summary()

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_batches,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_batches)

    if save_model:
        model.save('model/my_model.h5')
    learning_curves(history_fine)









