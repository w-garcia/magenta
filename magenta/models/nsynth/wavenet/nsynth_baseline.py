import csv
import argparse
import time

import numpy as np
import os
import tensorflow as tf
import utils
import fastgen

try:
    import cPickle as pickle
except:
    import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("source_path", "", "Path to directory with either "
                                              ".wav files or precomputed encodings in .npy files."
                                              "If .wav files are present, use wav files. If no "
                                              ".wav files are present, use .npy files")
tf.app.flags.DEFINE_string("npy_only", False, "If True, use only .npy files.")
tf.app.flags.DEFINE_string("save_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000",
                           "Path to checkpoint.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Number of samples per a batch.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")


class Model_Wrapper:
    def __init__(self, ckpt_path, audio_files):
        self.model = None
        self.sess = None
        self.seed = None
        self.audio = np.zeros([1, 1])
        self.audio_files = audio_files

        self.hop_length = fastgen.Config().ae_hop_length

        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Graph().as_default():
            sess = tf.Session(config=session_config)
            self.net = fastgen.load_fastgen_nsynth(batch_size=1)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)

            sess.run(self.net["init_ops"])
            self.sess = sess

    def get_probs(self, sample_i):
        enc_i = sample_i // self.hop_length
        all = self.sess.run(
            [self.net["predictions"], self.net["push_ops"]],
            feed_dict={self.net["X"]: self.audio, self.net["encoding"]: self.seed[:, enc_i, :]}
        )
        softmax = all[0]
        return softmax

    def get_index(self, sample):
        return sample

    def get_val(self, index):
        return index

    def get_seed(self):
        return self.seed

    def prediction(self, softmax):
        sample_bin = fastgen.sample_categorical(softmax)
        return sample_bin

    def set_seed(self, sample_length):
        self.seed = fastgen.load_batch(self.audio_files, sample_length=sample_length)

    def reset_placeholder(self):
        self.audio = np.zeros([1, 1])

    def calc_and_set_audio(self, sample_bin):
        # Expand companded bin
        new_audio = utils.inv_mu_law_numpy(sample_bin - 128)
        self.audio = new_audio
        return self.audio


def set_files():
    source_path = utils.shell_path(FLAGS.source_path)

    if tf.gfile.IsDirectory(source_path):
        files = tf.gfile.ListDirectory(source_path)
        postfix = ".npy"
        files = sorted([
            os.path.join(source_path, fname) for fname in files
            if fname.lower().endswith(postfix)
        ])
    elif source_path.lower().endswith((".npy")):
        files = [source_path]  # Pointed to 1 file only
    else:
        files = []

    return files


def main(unused_argv=None):
    checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
    save_path = utils.shell_path(FLAGS.save_path)
    if not save_path:
        raise RuntimeError("Must specify a save_path.")

    tf.logging.set_verbosity(FLAGS.log)

    model_wrap = Model_Wrapper(checkpoint_path, set_files())

    # in seconds
    out_sizes = [1, 2, 4, 8]
    for N in out_sizes:
        sample_length = N * 16000

        # This should pad according to what length we want
        model_wrap.set_seed(sample_length)

        # Some new encode length from .npy file, seems like 32 units = 1s
        encoding_length = model_wrap.seed.shape[1]
        batch_size = model_wrap.seed.shape[0]
        total_length = encoding_length * model_wrap.hop_length

        audio_batch = np.zeros((batch_size, total_length,), dtype=np.float32)

        # This is accessed during calls to tf graph
        model_wrap.reset_placeholder()

        # Only really have 1 file
        save_name = [os.path.join(save_path, "{}_".format(N) + os.path.splitext(os.path.basename(f))[0] + ".wav")
            for f in model_wrap.audio_files]

        start = time.clock()
        for sample_i in range(total_length):
            softmax = model_wrap.get_probs(sample_i)
            sample_bin = model_wrap.prediction(softmax)

            audio = model_wrap.calc_and_set_audio(sample_bin)
            audio_batch[:, sample_i] = audio[:, 0]

            if sample_i % 100 == 0:
                tf.logging.info("Sample: %d" % sample_i)
            if sample_i % 3000 == 0:
                fastgen.save_batch(audio_batch, save_name)

        sample_time = time.clock() - start
        print("Finish in {}".format(sample_time))
        fastgen.save_batch(audio_batch, save_name)


def console_entry_point():
    tf.app.run(main)


if __name__ == "__main__":
    console_entry_point()
