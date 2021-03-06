import time
import numpy as np
import string
import re
import os
import shutil
from typing import Dict, Tuple
import tensorflow as tf


def print_devices():
    visible_devices = tf.config.get_visible_devices()
    for devices in visible_devices:
        print(f"Visible device found: {devices}")


def mlp(vocab_size, embedding_dim, max_length, no_classes):
    sequence_input = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name="input0")
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
    x = embedding_layer(sequence_input)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(no_classes, activation='sigmoid', name="output0")(x)
    model = tf.keras.Model(sequence_input, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


def get_data_from_aclImdb(parameter: Dict) -> np.ndarray:
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    seed = 123
    raw_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=50, seed=seed)

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=parameter["vocab_size"],
        output_mode='int',
        output_sequence_length=parameter["sequence_length"])

    def vectorize_text_func(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = raw_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    raw_ds = raw_ds.map(vectorize_text_func)

    tokenized_texts = []
    for text, _ in raw_ds:
        tokenized_texts.append(text.numpy())

    return np.vstack(tokenized_texts)


def prepare_dataset(padded_input_data: np.ndarray, parameter: Dict) -> tf.data.Dataset:
    # to simulate multi-class problem, we randomly generate labels here
    # train_ds.map(lambda x, y: x, np.random.randint(low=0, high=no_classes - 1))
    labels = np.random.randint(low=0, high=parameter["no_classes"], size=(padded_input_data.shape[0]))
    category_labels_mat = tf.keras.utils.to_categorical(labels, num_classes=parameter["no_classes"])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(padded_input_data),
         tf.convert_to_tensor(category_labels_mat)))
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(parameter["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds


def run_test_track(padded_input_data: np.ndarray, parameter: Dict) -> Tuple[float, float]:
    train_ds = prepare_dataset(padded_input_data=padded_input_data, parameter=parameter)
    # build model
    model = mlp(vocab_size=parameter["vocab_size"], embedding_dim=parameter["embedding_dim"],
                max_length=parameter["sequence_length"], no_classes=parameter["no_classes"])

    # start training
    start_time = time.time()
    model.fit(train_ds, epochs=20, verbose=2)
    train_time = time.time() - start_time

    # start batch interference
    start_time = time.time()
    model.predict(train_ds, parameter["batch_size"])
    inference_time = (time.time() - start_time) / len(train_ds)
    return train_time, inference_time


def main():
    parameter = {"vocab_size": 80000,
                 "sequence_length": 150,
                 "embedding_dim": 100,
                 "batch_size": 1024}

    print_devices()
    padded_input_data = get_data_from_aclImdb(parameter=parameter)
    print("Load and prepare dataset")

    results = {"no classes": [], "training time": [], "inference time": []}
    for no_classes in [2, 50, 100, 10000, 20000]:
        parameter["no_classes"] = no_classes
        runtimes = run_test_track(padded_input_data=padded_input_data, parameter=parameter)
        results["no classes"].append(no_classes)
        results["training time"].append(runtimes[0])
        results["inference time"].append(runtimes[1])

    print(results)


if __name__ == '__main__':
    main()
