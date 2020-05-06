import os
import numpy as np
import tensorflow as tf
from imageio import imread
from utils import generate_barplot


def load_and_preprocess(filepath):
    img_size = 160
    image_data = imread(filepath)[:, :, :3]
    image_data = tf.cast(image_data, tf.float32)
    image_data = (image_data/127.5) - 1
    image_data = tf.image.resize(image_data, (img_size, img_size))
    image_data = np.expand_dims(np.array(image_data), axis=0)

    return tf.convert_to_tensor(image_data)


def get_model():
    file_path = f'{os.getcwd()}/model/my_model'
    if os.path.exists(file_path):
        model = tf.keras.models.load_model('model/my_model.h5')
    else:
        # TO DO : Get the model from online where I will upload
        model = tf.keras.models.load_model('model/my_model.h5')
        # pass
    return model


def predict_probabilities(image):
    model = get_model()
    prediction = model.predict(image, batch_size=1)
    prediction = tf.nn.softmax(prediction).numpy()
    prediction = np.reshape(prediction, prediction.shape[1])
    return prediction


if __name__ == '__main__':
    filepath = '/home/nishanth/PycharmProjects/image-classifier-webpage/images/12463.jpg'
    image = load_and_preprocess(filepath)
    prediction = predict_probabilities(image)
    script, div = generate_barplot(prediction, ['cat', 'dog'])
    print(prediction)