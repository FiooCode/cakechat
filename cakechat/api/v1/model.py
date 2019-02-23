import tensorflow as tf
import numpy as np
from random import randint
import datetime
import pickle
import os
from io import BytesIO
from tensorflow.python.lib.io import file_io
from tensorflow import __version__ as tf_version

if tf_version >= '1.1.0':
    MODE = 'rb'
else:  # for TF version 1.0
    MODE = 'r'
# Removes an annoying Tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def translateToSentences(inputs, wList, encoder=False):
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    numStrings = len(inputs[0])
    numLengthOfStrings = len(inputs)
    listOfStrings = [''] * numStrings
    for mySet in inputs:
        for index, num in enumerate(mySet):
            if (num != EOStokenIndex and num != padTokenIndex):
                if (encoder):
                    # Encodings are in reverse!
                    listOfStrings[index] = wList[num] + " " + listOfStrings[index]
                else:
                    listOfStrings[index] = listOfStrings[index] + " " + wList[num]
    listOfStrings = [string.strip() for string in listOfStrings]
    return listOfStrings


class ChatBot(object):
    def __init__(self, save_folder, word_list=None, buckets=False):
        """
        :param word_list: all words, obtained from pickled object.
        """
        self.buckets = buckets
        self.save_folder = save_folder
        self.cache_folder = self._make_dir(os.path.join(self.save_folder, "cache"))
        self.models_folder = self._make_dir(os.path.join(self.save_folder, "models"))
        self.log_folder = self._make_dir(os.path.join(self.save_folder, "logs"))
        self.dataset_folder = self._make_dir(os.path.join(self.save_folder, "dataset"))
        self.tensorboard_folder = self._make_dir(os.path.join(self.log_folder, "tensorboard"))
        # Default Hyperparamters
        self.batch_size = 24
        self.max_encoder_length = 15
        self.max_decoder_length = self.max_encoder_length
        self.lstm_units = 112
        self.embedding_dim = self.lstm_units
        self.num_layers_lstm = 3
        self.num_iterations = 500000
        self.word_list = word_list or self._get_word_list()
        self.vocab_size = len(self.word_list)
        tf.reset_default_graph()
        # Create the placeholders
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name="encoder_inputs") for i in
                               range(self.max_encoder_length)]
        self.decoder_labels = [tf.placeholder(tf.int32, shape=(None,), name="decoder_labels") for i in
                               range(self.max_decoder_length)]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name="decoder_inputs") for i in
                               range(self.max_decoder_length)]
        self.feed_previous = tf.placeholder(tf.bool, name="feed_previous")
        self.encoderLSTM = tf.contrib.rnn.BasicLSTMCell(self.lstm_units, state_is_tuple=True)
        # encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
        # Architectural choice of of whether or not to include ^
        self.decoder_outputs, self.decoder_final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoder_inputs,
            self.decoder_inputs,
            self.encoderLSTM,
            self.vocab_size,
            self.vocab_size,
            self.embedding_dim,
            feed_previous=self.feed_previous)
        self._sess = None
        # Define train variables
        self.decoder_prediction = tf.argmax(self.decoder_outputs, 2)
        self.loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in self.decoder_labels]
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs, self.decoder_labels,
                                                            self.loss_weights,
                                                            self.vocab_size)
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.zero_vector = np.zeros((1), dtype='int32')

    def _make_dir(self, dir):
        if not self.file_exists(dir):
            os.mkdir(dir)
            return dir
        return dir

    @classmethod
    def load_from(cls, meta_path, checkpoint_path):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        graph = tf.get_default_graph()
        opperations = graph.get_operations()
        encoder_inputs = opperations[0:15]
        decoder_labels = opperations[15:30]
        decoder_inputs = opperations[30:45]
        feed_previous = opperations[46]

    def _get_word_list(self):
        word_list_path = os.path.join(self.dataset_folder, "word_list.txt")
        if not self.file_exists(word_list_path):
            raise FileExistsError(
                "File word_list.txt was not found in: '{}'. Cannot continue...".format(word_list_path))
        word_list = pickle.load(self._get_file(word_list_path, "rb", get_pointer=True))
        # Need to modify the word list as well
        word_list.append('<pad>')
        word_list.append('<EOS>')
        return word_list

    def file_exists(self, path):
        if not self.buckets:
            does_it_exist = os.path.exists(path)
        else:
            does_it_exist = file_io.file_exists(path)
        return does_it_exist

    def _get_file(self, path, mode="r", get_pointer=False):
        modes = {"rb": "r+", "wb": "w+"}
        literal_mode = modes.get(mode, mode)
        if not self.buckets:
            if get_pointer:
                return open(path, mode)
            with open(path, mode) as f:
                file_data = f.read()
        else:
            f_stream = file_io.FileIO(path, literal_mode)
            if get_pointer:
                return f_stream
            file_data = BytesIO(f_stream.read())
        return file_data

    @property
    def sess(self):
        if not self._sess:
            self._sess = tf.Session()
        return self._sess

    @staticmethod
    def convert_ids_to_sentence(ids, w_list):
        """
        convert list of ids to a list of words (a sentence)
        :param ids: list- ids to convert.
        :param w_list: list- word list or vocabulary. (for converting)
        :return: list- converted words.
        """
        EOS_token_index = w_list.index('<EOS>')
        pad_token_index = w_list.index('<pad>')
        my_str = ""
        list_of_responses = []
        for num in ids:
            if num[0] == EOS_token_index or num[0] == pad_token_index:
                list_of_responses.append(my_str)
                my_str = ""
            else:
                my_str = my_str + w_list[num[0]] + " "
        if my_str:
            list_of_responses.append(my_str)
        list_of_responses = [i for i in list_of_responses if i]
        return list_of_responses

    @staticmethod
    def get_test_input(input_message, w_list, max_len):
        encoder_message = np.full((max_len), w_list.index('<pad>'), dtype='int32')
        input_split = input_message.lower().split()
        for index, word in enumerate(input_split):
            try:
                encoder_message[index] = w_list.index(word)
            except ValueError:
                continue
        encoder_message[index + 1] = w_list.index('<EOS>')
        encoder_message = encoder_message[::-1]
        encoder_message_list = []
        for num in encoder_message:
            encoder_message_list.append([num])
        return encoder_message_list

    def create_training_matrices(self, conversation_file_name, w_list, max_len):
        conversation_dictionary = np.load(self._get_file(conversation_file_name, "rb", get_pointer=True)).item()
        num_examples = len(conversation_dictionary)
        x_train = np.zeros((num_examples, max_len), dtype='int32')
        y_train = np.zeros((num_examples, max_len), dtype='int32')
        for index, (key, value) in enumerate(conversation_dictionary.items()):
            # Will store integerized representation of strings here (initialized as padding)
            encoder_message = np.full((max_len), w_list.index('<pad>'), dtype='int32')
            decoder_message = np.full((max_len), w_list.index('<pad>'), dtype='int32')
            # Getting all the individual words in the strings
            key_split = key.split()
            value_split = value.split()
            key_count = len(key_split)
            value_count = len(value_split)
            # Throw out sequences that are too long or are empty
            if (key_count > (max_len - 1) or value_count > (max_len - 1) or value_count == 0 or key_count == 0):
                continue
            # Integerize the encoder string
            for key_index, word in enumerate(key_split):
                try:
                    encoder_message[key_index] = w_list.index(word)
                except ValueError:
                    # TODO: This isnt really the right way to handle this scenario
                    encoder_message[key_index] = 0
            encoder_message[key_index + 1] = w_list.index('<EOS>')
            # Integerize the decoder string
            for value_index, word in enumerate(value_split):
                try:
                    decoder_message[value_index] = w_list.index(word)
                except ValueError:
                    decoder_message[value_index] = 0
            decoder_message[value_index + 1] = w_list.index('<EOS>')
            x_train[index] = encoder_message
            y_train[index] = decoder_message
        # Remove rows with all zeros
        y_train = y_train[~np.all(y_train == 0, axis=1)]
        x_train = x_train[~np.all(x_train == 0, axis=1)]
        num_examples = x_train.shape[0]
        return num_examples, x_train, y_train

    @staticmethod
    def get_training_batch(local_x_train, local_y_train, local_batch_size, max_len, num_training_examples, word_list):
        num = randint(0, num_training_examples - local_batch_size - 1)
        arr = local_x_train[num:num + local_batch_size]
        labels = local_y_train[num:num + local_batch_size]
        # Reversing the order of encoder string apparently helps as per 2014 paper
        reversed_list = list(arr)
        for index, example in enumerate(reversed_list):
            reversed_list[index] = list(reversed(example))
        # Lagged labels are for the training input into the decoder
        lagged_labels = []
        EOS_token_index = word_list.index('<EOS>')
        pad_token_index = word_list.index('<pad>')
        for example in labels:
            eos_found = np.argwhere(example == EOS_token_index)[0]
            shifted_example = np.roll(example, 1)
            shifted_example[0] = EOS_token_index
            # The EOS token was already at the end, so no need for pad
            if (eos_found != (max_len - 1)):
                shifted_example[eos_found + 1] = pad_token_index
            lagged_labels.append(shifted_example)
        # Need to transpose these
        reversed_list = np.asarray(reversed_list).T.tolist()
        labels = labels.T.tolist()
        lagged_labels = np.asarray(lagged_labels).T.tolist()
        return reversed_list, labels, lagged_labels

    def _get_training_matrices(self, conversation_file_name=None, w_list=None, max_len=None):
        x_train = self._np_array_cache("x_train.npy")
        y_train = self._np_array_cache("y_train.npy")

        if x_train is not None and y_train is not None:
            num_examples = x_train.shape[0]
        else:
            num_examples, x_train, y_train = self.create_training_matrices(conversation_file_name, w_list, max_len)
            np.save(os.path.join(self.cache_folder, "x_train.npy"), x_train)
            np.save(os.path.join(self.cache_folder, "y_train.npy"), y_train)
        print("Returning training matrices.")
        return num_examples, x_train, y_train

    def _np_array_cache(self, file_name, call_func=None):
        """
        If a numpy obj of the name given exists, then load it and return the data, otherwise it runs call_func(), saves it
        to the cache dir, then returns it.
        :param file_name: the file to check for.
        :param call_func: fallback if cache doesn't exist.
        :return: np array
        """
        file_path = os.path.join(self.cache_folder, file_name)
        if self.file_exists(file_path):
            print("Found cached files!")
            d = np.load(self._get_file(file_path, MODE, get_pointer=True))
            return d
        else:
            print("Couldn't find cached files...")
            if not call_func:
                return None
            ret_val = call_func()
            if ret_val:
                np.save(file_path, ret_val)
                return ret_val

    def load_model(self, load_from=None):
        load_from = load_from or self.models_folder
        if not self.file_exists(os.path.join(load_from, "checkpoint")):
            load_from = None
        if load_from:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(load_from))
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self, num_iterations, train_data_file, load_from=None):
        """
        Start training the model.
        :param num_iterations: int- number of iterations to train for.
        :param train_data_file: path- path to data file. (where train data is stored). If an array is found in np cache,
                                this will be ignored.
        :param load_from: path- Path to a checkpoint, if previously trained.
        """
        # Restore from checkpoint if provided.
        self.load_model(load_from)
        # Uploading results to Tensorboard
        tf.summary.scalar('Loss', self.loss)
        merged = tf.summary.merge_all()
        tb_dir = os.path.join(self.tensorboard_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = tf.summary.FileWriter(tb_dir, self.sess.graph)
        print("Getting training matrices.")

        num_training_examples, x_train, y_train = self._get_training_matrices(train_data_file, self.word_list,
                                                                              self.max_encoder_length)
        # Test data to use every so often when training.
        encoder_test_strings = ["Got any ideas?",
                                "hi",
                                "hi how are you",
                                "How do I fix this",
                                "Youre welcome"
                                ]
        print("TRAINING STARTED!")
        for i in range(num_iterations):
            # Prepare train data.
            encoder_train, decoder_target_train, decoder_input_train = self.get_training_batch(x_train, y_train,
                                                                                               self.batch_size,
                                                                                               self.max_encoder_length,
                                                                                               num_training_examples,
                                                                                               self.word_list)
            feed_dict = {self.encoder_inputs[t]: encoder_train[t] for t in range(self.max_encoder_length)}
            feed_dict.update({self.decoder_labels[t]: decoder_target_train[t] for t in range(self.max_decoder_length)})
            feed_dict.update({self.decoder_inputs[t]: decoder_input_train[t] for t in range(self.max_decoder_length)})
            feed_dict.update({self.feed_previous: False})
            # train.
            cur_loss, _, pred = self.sess.run([self.loss, self.optimizer, self.decoder_prediction], feed_dict=feed_dict)
            if i % 50 == 0:
                print('Current loss:', cur_loss, 'at iteration', i)
                summary = self.sess.run(merged, feed_dict=feed_dict)
                writer.add_summary(summary, i)
            if i % 25 == 0 and i != 0:
                random_num = randint(0, len(encoder_test_strings) - 1)
                print(encoder_test_strings[random_num])
                # Prepare test data.
                input_vector = self.get_test_input(encoder_test_strings[random_num], self.word_list,
                                                   self.max_encoder_length)
                feed_dict = {self.encoder_inputs[t]: input_vector[t] for t in range(self.max_encoder_length)}
                feed_dict.update({self.decoder_labels[t]: self.zero_vector for t in range(self.max_decoder_length)})
                feed_dict.update({self.decoder_inputs[t]: self.zero_vector for t in range(self.max_decoder_length)})
                feed_dict.update({self.feed_previous: True})
                # Run test and print results.
                ids = (self.sess.run(self.decoder_prediction, feed_dict=feed_dict))
                print(self.convert_ids_to_sentence(ids, self.word_list))
            if i % 10000 == 0 and i != 0:
                # Save checkpoint every 10000 iterations.
                checkpoint = os.path.join(self.models_folder, "pretrained_seq2seq.ckpt")
                save_path = self.saver.save(self.sess, checkpoint, global_step=i)

    def save_model(self, out_path=None):
        out_path = out_path or os.path.join(self.models_folder, "saved_model")
        inputs = {"encoder_inputs" + str(t): self.encoder_inputs[t] for t in range(self.max_encoder_length)}
        inputs.update({"decoder_labels" + str(t): self.decoder_labels[t] for t in range(self.max_decoder_length)})
        inputs.update({"decoder_inputs" + str(t): self.decoder_inputs[t] for t in range(self.max_decoder_length)})
        inputs.update({"feed_previous": self.feed_previous})
        tf.saved_model.simple_save(
            self.sess,
            out_path,
            inputs=inputs,
            outputs={self.decoder_outputs[t].name: self.decoder_outputs[t] for t in range(self.max_decoder_length)}
        )
        builder = tf.saved_model.builder.SavedModelBuilder(out_path)
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            main_op=tf.tables_initializer())
        builder.save()

    def make_prediction(self, input_text):
        input_vector = self.get_test_input(input_text, self.word_list, self.max_encoder_length)
        # Prepare test data.
        feed_dict = {self.encoder_inputs[t]: input_vector[t] for t in range(self.max_encoder_length)}
        feed_dict.update({self.decoder_labels[t]: self.zero_vector for t in range(self.max_decoder_length)})
        feed_dict.update({self.decoder_inputs[t]: self.zero_vector for t in range(self.max_decoder_length)})
        feed_dict.update({self.feed_previous: True})
        # Run test and print results.
        ids = (self.sess.run(self.decoder_prediction, feed_dict=feed_dict))
        return self.convert_ids_to_sentence(ids, self.word_list)


def run_train(top_dir, num_training_steps=500000, load_from=None):
    npy_train_data_path = os.path.join(top_dir, "dataset", "traindata.npy")
    path_to_wordlist = os.path.join(top_dir, "dataset", "word_list.txt")
    chat_bot = ChatBot(top_dir)
    chat_bot.train(num_training_steps, npy_train_data_path, load_from=load_from)


class ChatBotApi(object):
    def __init__(self, word_list_path, buckets=False):
        self.buckets = buckets
        self.word_list = self._get_word_list(word_list_path)

    def _get_word_list(self, word_list_path):
        if not self.file_exists(word_list_path):
            raise FileExistsError(
                "File word_list.txt was not found in: '{}'. Cannot continue...".format(word_list_path))
        word_list = pickle.load(self._get_file(word_list_path, "rb", get_pointer=True))
        # Need to modify the word list as well
        word_list.append('<pad>')
        word_list.append('<EOS>')
        return word_list

    def file_exists(self, path):
        if not self.buckets:
            does_it_exist = os.path.exists(path)
        else:
            does_it_exist = file_io.file_exists(path)
        return does_it_exist

    def _get_file(self, path, mode="r", get_pointer=False):
        modes = {"rb": "r+", "wb": "w+"}
        literal_mode = modes.get(mode, mode)
        if not self.buckets:
            if get_pointer:
                return open(path, mode)
            with open(path, mode) as f:
                file_data = f.read()
        else:
            f_stream = file_io.FileIO(path, literal_mode)
            if get_pointer:
                return f_stream
            file_data = BytesIO(f_stream.read())
        return file_data

    def create_input(self, message, max_encoder_length=15, max_decoder_length=15):
        zero_vector = np.zeros((1), dtype='int32')
        word_list = ChatBot
        input_vector = ChatBot.get_test_input(message, self.word_list, max_encoder_length)
        inputs = {"encoder_inputs" + str(t): input_vector[t] for t in range(max_encoder_length)}
        inputs.update({"decoder_labels" + str(t): zero_vector[t] for t in range(max_decoder_length)})
        inputs.update({"decoder_inputs" + str(t): zero_vector[t] for t in range(max_decoder_length)})
        inputs.update({"feed_previous": True})
        return inputs


if __name__ == "__main__":
    print("STARTING!")
    SAVE_PATH = "save_path"
    BUCKET = '/home/marc/Mass_Storage_1TB/nextCloud/Business/Upwork/Projects/Fredrik_Gabriel/ChatBot'
    INPUT_DIR = os.path.join(BUCKET, SAVE_PATH)
    TRAIN_FILE = os.path.join(INPUT_DIR, "dataset/train_dict.npy")
    # out_folder = os.path.join(BUCKET, SAVE_PATH)
    print("Finished Loading Input.")
    # run_train(INPUT_DIR, num_training_steps=500000, load_from="/home/marc/Mass_Storage_1TB/nextCloud/Business/Upwork/Projects/Fredrik_Gabriel/ChatBot/save_path/models")
    chatbot = ChatBot(INPUT_DIR)
