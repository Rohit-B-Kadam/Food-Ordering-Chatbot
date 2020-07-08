import re

import sys
import os
import json
import tensorflow as tf
from numpy.core.multiarray import ndarray


from path_manager import PATHS
from model.training.transformer_model import transformer, CustomSchedule

UTTERANCES_FILE = PATHS["slot_tagger_dataset"]["utterances"]
BIO_TAGS_FILE = PATHS["slot_tagger_dataset"]["bio_tags"]
INTENT_FILE = PATHS["slot_tagger_dataset"]["intent"]
MODEL_CHECKPOINTS_PATH = "model/training/weights/checkpoints"
TOKENIZER_PATH = "model/training/weights/tokenizer.json"

START_TOKEN = "BOS "    # BOS: Begin of sentence
END_TOKEN = " EOS"      # EOS: End of sentence

# Data Parameter
BATCH_SIZE = 64
BUFFER_SIZE = 200   # range in which data will shuffle (2000)
VOCAB_SIZE = 0      # while after training tokenizer we setting this value
MAX_LENGTH = 128    # Maximum sentence length (512)

# Model parameters
NUM_LAYERS = 4      # Number of encoder layer and decoder layer in ENCODER and DECODER
D_MODEL = 256       # Dense Model units
NUM_HEADS = 8       # Number of head in Multi head attention
UNITS = 512         #
DROPOUT = 0.1       #
EPOCHS = 2          #


def load_data(utterances_file_path: str, tags_file_path: str, intent_file_path: str):
    utterances = None
    tags = None
    intents = None

    with open(utterances_file_path, "r") as fd:
        utterances = fd.readlines()

    with open(tags_file_path, "r") as fd:
        tags = fd.readlines()

    with open(intent_file_path, "r") as fd:
        intents = fd.readlines()

    return utterances, tags, intents


def preprocess_utterance(utterance: str) -> str:
    utterance = utterance.lower().strip()  # convert to lowercase and remove trailing space
    utterance = START_TOKEN + utterance + END_TOKEN  # adding start and end token
    return utterance


def preprocess_tags(tag):
    # adding start and end token
    tag = START_TOKEN + tag.strip() + END_TOKEN
    return tag


def train_save_tokenizer(texts, tags):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    tokenizer.fit_on_texts(texts + tags)  # train tokenizer

    text_tensor = tokenizer.texts_to_sequences(texts)  # convert text to numerical token
    text_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        text_tensor, maxlen=MAX_LENGTH, padding='post')  # pad all tensor

    tag_tensor = tokenizer.texts_to_sequences(tags)  # convert text to numerical token
    tag_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tag_tensor, maxlen=MAX_LENGTH, padding='post')  # pad all tensor

    # save tokenizer
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # setting VOCAB_SIZE
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer.word_counts) + 1

    return text_tensor, tag_tensor, tokenizer


def get_save_tokenizer():

    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    # setting VOCAB_SIZE
    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer.word_counts) + 1
    return tokenizer


def create_tf_dataset(utterances: ndarray, tags: ndarray):
    # Teacher Forcing technique
    # Decoder inputs use the previous target as input
    # remove START_TOKEN from targets

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': utterances,
            'dec_inputs': tags[:, :-1]  # remove last word
        },
        {
            'outputs': tags[:, 1:]  # removing first word ie start-token
        },
    ))

    # dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    # dataset = dataset.cache()
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Loss Function
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def train_transformer(dataset):
    tf.keras.backend.clear_session()

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model.summary()
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = MODEL_CHECKPOINTS_PATH + "/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=100)

    model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback], validation_split=0.2)

    return model


def train_model():
    print("Training the model...")
    utterances, tags, intents = load_data(UTTERANCES_FILE, BIO_TAGS_FILE, INTENT_FILE)
    print(f"Dataset size: {len(utterances)}")

    utterances = list(map(preprocess_utterance, utterances))
    tags = list(map(preprocess_tags, tags))

    utterance_tensor, tag_tensor, tokenizer = train_save_tokenizer(utterances, tags)
    print(f"Max sentence length: {len(utterance_tensor[0])}")
    print(f"Utterance Tokenized sample:  {utterance_tensor[10]}")
    print(f"Max tags sequence length: {len(tag_tensor[0])}")
    print(f"Tags tokenized sample: {tag_tensor[10]}")

    dataset = create_tf_dataset(utterance_tensor, tag_tensor)
    model = train_transformer(dataset)

    return model


def evaluate(sentence, tokenizer, model):
    sentence = preprocess_utterance(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence, tokenizer, model):
    prediction = evaluate(sentence, tokenizer, model)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


def evaluate_model():
    print("Evaluating the model...")
    tokenizer = get_save_tokenizer()
    print(tokenizer.fit_on_texts("Hi , how are you ?"))


def print_help():
    print("Provide following command line argument: ")
    print("--train : for training")
    print("--eval  : for evaluate")


def main():
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "--train":
            train_model()
        elif sys.argv[1].lower() == "--eval":
            evaluate_model()
        else:
            print_help()
    else:
        print_help()


if __name__ == "__main__":
    main()
