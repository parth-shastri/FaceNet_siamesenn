# TODO: DON'T MESS UP NAMES OF VARIABLES
# TODO: prefer the SavedModel method for saving instead of saving in the hdf5 format if custom objects are present
import tensorflow as tf
from keras.applications import resnet
import numpy as np
import matplotlib.pyplot as plt
from keras import applications
from tensorflow.keras import layers
from tensorflow.keras import metrics, Model, optimizers, losses
import os
import json
import config


def preprocess(image_path):

    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=config.IMAGE_SHAPE)

    return image


def preprocess_triplets(anchor, pos, neg):

    return (
        preprocess(anchor),
        preprocess(pos),
        preprocess(neg)
    )


anchor_images = sorted([os.path.join(config.ANCHOR_DIR, p) for p in os.listdir(config.ANCHOR_DIR)])
pos_images = sorted([os.path.join(config.POS_DIR, p) for p in os.listdir(config.POS_DIR)])

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
pos_dataset = tf.data.Dataset.from_tensor_slices(pos_images)

rng = np.random.default_rng(232)
rng.shuffle(anchor_images)
rng.shuffle(pos_images)

neg_images = anchor_images + pos_images


neg_dataset = tf.data.Dataset.from_tensor_slices(neg_images)
neg_dataset = neg_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, pos_dataset, neg_dataset))
dataset = dataset.map(preprocess_triplets).shuffle(buffer_size=1024)

train_data = dataset.take(round(image_count * 0.8)).batch(config.BATCH_SIZE).prefetch(8)
val_data = dataset.skip(round(image_count * 0.8)).batch(config.BATCH_SIZE).prefetch(8)


def visualize(anchor, pos, neg):
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        ax1 = axs[i, 0]
        ax1.imshow(anchor[i])
        ax2 = axs[i, 1]
        ax2.imshow(pos[i])
        ax3 = axs[i, 2]
        ax3.imshow(neg[i])

    plt.show()


def get_base_extractor():
    be = resnet.ResNet50(include_top=False,
                        weights='imagenet',
                        input_shape=config.IMAGE_SHAPE + (3,))
    for layer in be.layers:
        if layer.name == "conv5_block1_out":
            layer.trainable = True

        layer.trainable = False
    return be


class Embedding(layers.Layer):
    def __init__(self, embed_dim=256):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.base_feature_extractor = get_base_extractor()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(256, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.out = layers.Dense(self.embed_dim, name='output_dense')

    def call(self, inputs, **kwargs):
        x = self.base_feature_extractor(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        out = self.out(x)

        return out

    """ good practice to define get_config and from config in case of custom subclassing"""

    def get_config(self):  # to save this model in serialized .h5 format we have to override the get_config method
        config = super(Embedding, self).get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

    @classmethod  # No need to call this as this is in the same format by default (just for understanding purpose
    def from_config(cls, config):
        return cls(**config)


class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, anchor, positive, negative):

        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance

    """ good practice to define get_config and from config in case of custom subclassing"""
    def get_config(self):   # to save this model in serialized .h5 format we have to override the get_config method
        config = super(DistanceLayer, self).get_config()
        return config

    @classmethod     # No need to call this as this is in the same format by default (just for understanding purpose
    def from_config(cls, config):
        return cls(**config)


class FaceNet(Model):

    def __init__(self, margin=0.5, embed_dim=256):
        super(FaceNet, self).__init__()
        self.margin = margin
        self.embed_dim = embed_dim
        self.embedding = Embedding(embed_dim=self.embed_dim)
        self.distance = DistanceLayer()
        self.loss_tracker = tf.metrics.Mean(name='loss')

    def train_step(self, data):

        with tf.GradientTape(persistent=True) as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.embedding.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.embedding.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs, training=None, mask=None):

        anchor_in, positive_in, negative_in = inputs
        distances = self.distance(
            self.embedding(resnet.preprocess_input(anchor_in)),
            self.embedding(resnet.preprocess_input(positive_in)),
            self.embedding(resnet.preprocess_input(negative_in)),
        )
        return distances

    def _compute_loss(self, data):
        ap_dist, an_dist = self.call(data)

        loss = ap_dist - an_dist
        loss = tf.maximum(loss + self.margin, 0)

        return loss

    @property
    def metrics(self):

        return [self.loss_tracker]

    """ good practice to define get_config and from config in case of custom subclassing"""
    def get_config(self):   # to save this model in serialized .h5 format we have to override the get_config method
        config = {"margin": self.margin, "embed_dim": self.embed_dim}
        return config

    @classmethod    # No need to call this as this is in the same format by default (just for understanding purpose
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


tr_model = FaceNet(margin=config.MARGIN, embed_dim=config.EMBED_DIM)
tr_model.compile(optimizer=optimizers.Adam(0.0001))

ckpt = tf.keras.callbacks.ModelCheckpoint(config.CKPT_DIR, save_freq='epoch')  # saves after each epoch
# model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[ckpt])
json_config = tr_model.to_json()

with open("model_architecture.json", "w") as fp:
    json.dump(json_config, fp)

# must be provided in order to deserialize
# TODO: Don't mess up names while assigning them in the overridden 'get_config' method in custom Model
custom_objects = {"FaceNet": FaceNet, "DistanceLayer": DistanceLayer, "resnet": resnet}
model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)

model.load_weights(config.CKPT_DIR)

if __name__ == "__main__":

    '''calculation of the similarity (Cosine Similarity)'''

    samples = next(iter(val_data))
    anchor, pos, neg = samples
    anchor_embedding = model.embedding(resnet.preprocess_input(anchor))
    pos_embedding = model.embedding(resnet.preprocess_input(pos))
    neg_embedding = model.embedding(resnet.preprocess_input(neg))

    print(f"The shape of embeddings :{anchor_embedding.shape, pos_embedding.shape, neg_embedding.shape}")

    similarity = metrics.CosineSimilarity()

    pos_sim = similarity(anchor_embedding, pos_embedding)
    neg_sim = similarity(anchor_embedding, neg_embedding)

    assert pos_sim.numpy() > neg_sim.numpy()

    print(f"The similarity between positive image and anchor:{pos_sim}")
    print(f"The similarity between negative image and anchor:{neg_sim}")


