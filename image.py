import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

class Dataset:
    def __init__(self, images_path = None, annotation_csv = None, image_size=(224,224), batch_size=16):
        if not images_path or not annotation_csv:
            return

        
        self.images_path = images_path
        self.annotation_csv = annotation_csv
        self.image_size = image_size
        self.batch_size = batch_size

        self.df = pd.read_csv(annotation_csv)

        self.df = self.df.groupby('filename').first().reset_index()
        self.size = len(self.df)

        self.filenames = self.df['filename'].tolist()

        # print(self.filenames)

        # Scale the boundign boxes
        y_scale = 1 / self.df['height']
        x_scale = 1 / self.df['width']
        self.df['xmin'] = x_scale * self.df['xmin']
        self.df['xmax'] = x_scale * self.df['xmax']
        self.df['ymin'] = y_scale * self.df['ymin']
        self.df['ymax'] = y_scale * self.df['ymax']
        self.boundign_box = self.df[['xmin','ymin','xmax','ymax']].values.astype('float32')
        # print(self.boundign_box)

    def preprocess_image(self, filename):
        img_path = os.path.join(self.images_path, filename)
        img = load_img(img_path, target_size=self.image_size)
        img = tf.convert_to_tensor(img_to_array(img) / 255.0, dtype=tf.float32)

        # print(img.shape)
        return img

    def load(self):
        # images = []
        # for f in tqdm(self.filenames):
        #     images = images + [self.preprocess_image(f)]
        # images = tf.convert_to_tensor(images)

        # bboxes = tf.convert_to_tensor(self.boundign_box)

        # dataset = tf.data.Dataset.from_tensor_slices((images, bboxes))
        # dataset = dataset.shuffle(len(images)).batch(self.batch_size)
        # return dataset

        def gen():
            for f, bbox in zip(self.filenames, self.boundign_box):
                img = self.preprocess_image(f)
                bbox = tf.convert_to_tensor(bbox)
                yield img, bbox

        output_signature = (
            tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        dataset = dataset.shuffle(len(self.filenames)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

