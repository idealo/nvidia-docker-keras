import time
import numpy as np
import string
import re
import os
import sys
import shutil
from typing import Dict, Tuple
import tensorflow as tf
from official.nlp import optimization
import tensorflow_hub as hub
import tensorflow_text as text


def print_devices() -> None:
    """
    Print number of gpu devices to be used
    """
    print("\n------------------------")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("------------------------\n")


def mlp(vocab_size : int, embedding_dim : int, max_length : int, no_classes : int) -> tf.keras.Model:
    """
    Build multi-layer perceptron model

    :param int vocab_size: vocabulary size
    :param int embedding_dim: embedding size
    :param int max_length: maximal length of the padded sequence
    :param int no_classes: number of classes / output layer size
    :return:  model object
    :rtype: tf.keras.Model
    """
    sequence_input = tf.keras.layers.Input(shape=(max_length,), dtype='int32', name="input0")
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
    x = embedding_layer(sequence_input)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(no_classes, activation='sigmoid', name="output0")(x)
    model = tf.keras.Model(sequence_input, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model


def bert(train_ds : tf.data.Dataset, epochs : int, no_classes) -> tf.keras.Model:
    """
    Build bert model

    :param tf.data.Dataset train_ds: training dataset
    :param int epochs: no epochs
    :param int no_classes: number of classes / output layer size
    :return: model object
    :rtype: tf.keras.Model
    """
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    x = tf.keras.layers.Dense(512, activation='relu')(net)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(no_classes, activation='sigmoid', name="output0")(x)
    model = tf.keras.Model(text_input, output)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.metrics.BinaryAccuracy()
    model.compile()
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    model.summary()
    return model


def custom_standardization(input_data):
    """
    Function of standardizing text

    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


def get_data_from_aclImdb() -> tf.data.Dataset:
    """
    Load aclImdb_v1 dataset from internet

    :return: dataset object
    :rtype: tf.data.Dataset
    """
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
    return raw_ds


def preprocess_mlp_text(dataset : tf.data.Dataset, parameter: Dict) -> np.ndarray:
    """
    Perform tokenization for MLP model

    :param tf.data.Dataset dataset: dataset containing text and label data
    :param Dict parameter: parameter object containing vocab_size and sequence_lenght parameter
    :return: tokenized and padded text
    :rtype: np.ndarray
    """
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=parameter["vocab_size"],
        output_mode='int',
        output_sequence_length=parameter["sequence_length"])

    def vectorize_text_func(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    text_ds = dataset.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    text_ds = dataset.map(vectorize_text_func)

    tokenized_texts = []
    for text, _ in text_ds:
        tokenized_texts.append(text.numpy())

    return np.vstack(tokenized_texts)


def preprocess_bert_text(dataset : tf.data.Dataset) -> np.ndarray:
    """
    Perform tokenization for BERT model

    :param tf.data.Dataset dataset: dataset containing text and label data
    :return: tokenized and padded text
    :rtype: np.ndarray
    """

    text_ds = dataset.map(lambda x, y: x)
    tokenized_texts = []
    for text in text_ds:
        tokenized_texts += text.numpy().tolist()

    return np.vstack(tokenized_texts)


def prepare_dataset(text_data: np.ndarray, parameter: Dict, no_samples : int) -> tf.data.Dataset:
    """
    To simulate multi-class problem, we randomly generate labels here

    :param np.ndarray text_data: text data
    :param Dict parameter: parameter object
    :param int no_samples: dataset size
    :return: dataset object
    :rtype:  tf.data.Dataset
    """
    print(f"\nDataset contains {no_samples} samples\n")
    labels = np.random.randint(low=0, high=parameter["no_classes"], size=(no_samples))
    category_labels_mat = tf.keras.utils.to_categorical(labels, num_classes=parameter["no_classes"])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(text_data),
         tf.convert_to_tensor(category_labels_mat)))
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(parameter["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds


def run_mlp_test_track(train_ds: np.ndarray, parameter: Dict) -> Tuple[float, float]:

    # build model
    print("create mlp model")
    model = mlp(vocab_size=parameter["vocab_size"], embedding_dim=parameter["embedding_dim"],
                max_length=parameter["sequence_length"], no_classes=parameter["no_classes"])
    print("complete")
    # start training
    print("start training")
    start_time = time.time()
    model.fit(train_ds, epochs=parameter["epochs"], verbose=2)
    train_time = time.time() - start_time
    print(f"complete in {train_time} [sec]")
    # start batch interference
    print("start inference test")
    start_time = time.time()
    model.predict(train_ds)
    inference_time = (time.time() - start_time) / len(train_ds)
    print(f"complete in {inference_time} [sec]")
    return train_time, inference_time


def run_bert_test_track(train_ds: tf.data.Dataset, parameter: Dict) -> Tuple[float, float]:
    # build model
    print("create mlp model")
    model = bert(train_ds=train_ds, epochs=parameter["epochs"], no_classes=parameter["no_classes"])
    print("complete")
    # start training
    print("train model")
    start_time = time.time()
    model.fit(train_ds, epochs=parameter["epochs"], verbose=2)
    train_time = time.time() - start_time
    print(f"complete in {train_time} [sec]")
    # start batch interference
    print("start inference test")
    start_time = time.time()
    model.predict(train_ds)
    inference_time = (time.time() - start_time) / len(train_ds)
    print(f"complete in {inference_time} [sec]")
    return train_time, inference_time


def main():
    parameter = {"vocab_size": 80000,
                 "sequence_length": 150,
                 "embedding_dim": 100,
                 "batch_size": 128,
                 "epochs" : 5}

    print_devices()
    train_dataset = get_data_from_aclImdb()
    mlp_text_data = preprocess_mlp_text(dataset=train_dataset, parameter=parameter)
    bert_text_data = preprocess_bert_text(dataset=train_dataset)
    print("Load and prepare dataset")

    results ={"mlp" : {"no classes": [], "training time": [], "inference time": []},
              "bert" : {"no classes": [], "training time": [], "inference time": []}}

    for no_classes in [2, 50, 100, 10000, 20000]:
        parameter["no_classes"] = no_classes
        train_ds = prepare_dataset(text_data=mlp_text_data, parameter=parameter, no_samples=mlp_text_data.shape[0])
        runtimes = run_mlp_test_track(train_ds=train_ds, parameter=parameter)
        results["mlp"]["no classes"].append(no_classes)
        results["mlp"]["training time"].append(runtimes[0])
        results["mlp"]["inference time"].append(runtimes[1])


        train_ds = prepare_dataset(text_data=bert_text_data, parameter=parameter, no_samples=bert_text_data.shape[0])
        runtimes = run_bert_test_track(train_ds=train_ds, parameter=parameter)
        results["bert"]["no classes"].append(no_classes)
        results["bert"]["training time"].append(runtimes[0])
        results["bert"]["inference time"].append(runtimes[1])

    print(results)


if __name__ == '__main__':
    main()
