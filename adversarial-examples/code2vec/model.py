from itertools import islice

import tensorflow as tf

import PathContextReader
import numpy as np
import time
import pickle
import random
from common import common, VocabType
import adversarialsearcher
from adversarialsearcher import AdversarialSearcher, AdversarialTargetedSearcher, \
    AdversarialSearcherTrivial, AdversarialTargetedSearcherTrivial, \
    AdversarialSearcherTfidf, AdversarialTargetedSearcherTfidf
import codeguard
from codeguard import guard_by_n2p, guard_by_vunk
import common_adversarial
import re


def my_py_func(grad_word_embed, path_source_target_tensor, words_to_compute_grads, partial_words_vocab):
    mask = tf.equal(words_to_compute_grads, path_source_target_tensor)
    grad_word_embed = tf.reduce_sum(tf.multiply(grad_word_embed, tf.expand_dims(tf.cast(mask, tf.float32), -1)), axis=1)

    grad_of_input = tf.matmul(grad_word_embed, partial_words_vocab, transpose_b=True)

    return grad_of_input

class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.eval_data_lines = None
        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op = None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None

        # TODO noam: loss&gradient w.r.t. input
        self.loss_wrt_input, self.grad_wrt_input, self.adversarial_name, \
            self.adversarial_name_index = None, None, None, None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            with open('{}.dict.c2v'.format(config.TRAIN_PATH), 'rb') as file:
                word_to_count = pickle.load(file)
                path_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')
            self.config.NUM_EXAMPLES = num_training_examples

            self.word_to_index, self.index_to_word, self.word_vocab_size = \
                common.load_vocab_from_dict(word_to_count, config.WORDS_VOCAB_SIZE, start_from=1)
            print('Loaded word vocab. size: %d' % self.word_vocab_size)

            self.target_word_to_index, self.index_to_target_word, self.target_word_vocab_size = \
                common.load_vocab_from_dict(target_to_count, config.TARGET_VOCAB_SIZE,
                                            start_from=1)
            print('Loaded target word vocab. size: %d' % self.target_word_vocab_size)

            self.path_to_index, self.index_to_path, self.path_vocab_size = \
                common.load_vocab_from_dict(path_to_count, config.PATHS_VOCAB_SIZE,
                                            start_from=1)
            print('Loaded paths vocab. size: %d' % self.path_vocab_size)

        self.create_index_to_target_word_map()

    def create_index_to_target_word_map(self):
        self.index_to_target_word_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(list(self.index_to_target_word.keys()),
                                                            list(self.index_to_target_word.values()),
                                                            key_dtype=tf.int64, value_dtype=tf.string),
                default_value=tf.constant(common.noSuchWord, dtype=tf.string))

    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_evaluate = max(int(
            self.config.NUM_EXAMPLES / self.config.BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS), 1)

        self.queue_thread = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                path_to_index=self.path_to_index,
                                                                target_word_to_index=self.target_word_to_index,
                                                                config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.input_tensors())
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)
        with self.queue_thread.start(self.sess):
            time.sleep(1)
            print('Started reader...')
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        print('Number of waiting examples in queue: %d' % self.sess.run(
                            "shuffle_batch/random_shuffle_queue_Size:0"))
                        sum_loss = 0
                        multi_batch_start_time = time.time()
                    if batch_num % num_batches_to_evaluate == 0:
                        epoch_num = int((batch_num / num_batches_to_evaluate) * self.config.SAVE_EVERY_EPOCHS)
                        save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                        self.save_model(self.sess, save_target)
                        print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                        results, precision, recall, f1 = self.evaluate()
                        print('Accuracy after %d epochs: %s' % (epoch_num, results[:5]))
                        print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                            recall) + ', F1: ' + str(f1))
            except tf.errors.OutOfRangeError:
                print('Done training')

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def evaluate(self, guard_input=None):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                  path_to_index=self.path_to_index,
                                                                  target_word_to_index=self.target_word_to_index,
                                                                  config=self.config, is_evaluating=True)
            self.eval_placeholder = self.eval_queue.get_input_placeholder()
            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, _, _, _ = \
                self.build_test_graph(self.eval_queue.get_filtered_batches())
            self.saver = tf.train.Saver()

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name )
                self.saver.save(self.sess, release_name )
                return None

        if self.eval_data_lines is None:
            print('Loading test data from: ' + self.config.TEST_PATH)
            self.eval_data_lines = common.load_file_lines(self.config.TEST_PATH)
            print('Done loading test data')

        if guard_input is not None:
            word_embeddings = self.get_words_vocab_embed()
            print("Guard input is active. (make sure dataset includes variables-list)")

        with open('log.txt', 'w') as output_file:
            num_correct_predictions = np.zeros(self.topk)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            start_time = time.time()

            measure_by_percentage = {}

            for batch in common.split_to_batches(self.eval_data_lines, self.config.TEST_BATCH_SIZE):
                original_batch = batch
                if guard_input is not None:
                    # TODO: debug this
                    batch = self.guard_code_batch(batch, word_embeddings, guard_input)

                top_words, top_scores, original_names = self.sess.run(
                    [self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op],
                    feed_dict={self.eval_placeholder: batch})
                top_words, original_names = common.binary_to_string_matrix(top_words), common.binary_to_string_matrix(
                    original_names)
                # Flatten original names from [[]] to []
                original_names = [w for l in original_names for w in l]

                num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                          zip(original_names, top_words))
                true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                    zip(original_names, top_words),
                    true_positive, false_positive, false_negative)

                total_predictions += len(original_names)
                total_prediction_batches += 1
                if total_prediction_batches % self.num_batches_to_log == 0:
                    elapsed = time.time() - start_time
                    # start_time = time.time()
                    self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed, len(self.eval_data_lines))

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')


        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        del self.eval_data_lines
        self.eval_data_lines = None

        with open("per.pkl", "wb") as f:
            pickle.dump(measure_by_percentage, f)

        return num_correct_predictions / total_predictions, precision, recall, f1

    def adversarial_training(self):
        topk_words_from_model = 100
        word_to_indextop, indextop_to_word  = self.create_ordered_words_dictionary(
            self.config.TRAIN_PATH,
            self.config.MAX_WORDS_FROM_VOCAB_FOR_ADVERSARIAL)

        print('Starting adversarial training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_evaluate = max(int(
            self.config.NUM_EXAMPLES / self.config.BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS), 1)

        self.adversarial_training_queue_thread = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                  path_to_index=self.path_to_index,
                                                                  target_word_to_index=self.target_word_to_index,
                                                                  config=self.config, is_evaluating=True)
        self.adversarial_training_placeholder = self.adversarial_training_queue_thread.get_input_placeholder()

        # create test graph
        self.train_placeholder = self.adversarial_training_placeholder
        # self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, \
        # self.eval_source_string, _, self.eval_path_target_string = \
        #     self.build_test_graph(self.adversarial_training_queue_thread.get_filtered_batches())

        # create adversarial training graph
        self.words_to_compute_grads = tf.placeholder(tf.string, [None])

        self.loss_wrt_input, self.grad_wrt_input, self.adversarial_name, self.adversarial_name_index, \
        adversarial_optimizer, adversarial_optimizer_train_loss = \
            self.build_adversarial_training_graph_with_loss(self.adversarial_training_queue_thread.get_filtered_batches(),
                                      self.adversarial_training_queue_thread,
                                            indextop_to_word, self.words_to_compute_grads, topk_words_from_model)


        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

        self.initialize_session_variables(self.sess)

        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        path = self.config.TRAIN_PATH + '.train.c2v'
        epoch = 0
        while True:
            epoch += 1
            print("EPOCH:", epoch)
            with open(path, 'r') as f:
                for batch_lines in iter(lambda: tuple(islice(f, self.config.BATCH_SIZE)), ()):

                    batch_lines = [l.rstrip('\n') for l in batch_lines]
                    batch_num += 1

                    # regular training
                    batch_data = [common_adversarial.separate_vars_code(l)[1] for l in batch_lines]
                    _, batch_loss = self.sess.run([adversarial_optimizer, adversarial_optimizer_train_loss],
                                                  feed_dict={self.train_placeholder: batch_data})
                    sum_loss += batch_loss

                    # adversarial training
                    batch_searchers = [AdversarialSearcher(1, 1, word_to_indextop, indextop_to_word,
                                                           line, None)
                                       for line in batch_lines]
                    batch_searchers = [se for se in batch_searchers if se.can_be_adversarial()]

                    if batch_searchers:
                        [se.pop_unchecked_adversarial_code() for se in batch_searchers]

                        batch_data = [se.get_adversarial_code() for se in batch_searchers]
                        batch_word_to_derive = [se.get_word_to_derive() for se in batch_searchers]
                        loss_of_input, grad_of_input = self.sess.run([self.loss_wrt_input,self.grad_wrt_input],
                            feed_dict={self.train_placeholder: batch_data, self.words_to_compute_grads: batch_word_to_derive})
                        for searcher, grads in zip(batch_searchers, grad_of_input):
                            searcher.next((0, "", grads))

                        batch_data = [c for se in batch_searchers
                                      for _, c in se.pop_unchecked_adversarial_code()]
                        _, batch_loss = self.sess.run([adversarial_optimizer, adversarial_optimizer_train_loss],
                                                      feed_dict={self.train_placeholder: batch_data})
                        sum_loss += batch_loss

                    # logging
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        print('batch num: {}/{}'.format(batch_num, num_batches_to_evaluate))
                        # print('Number of waiting examples in queue: %d' % self.sess.run(
                        #     "shuffle_batch/random_shuffle_queue_Size:0"))
                        sum_loss = 0
                        multi_batch_start_time = time.time()
                    if batch_num % num_batches_to_evaluate == 0:
                        epoch_num = int((batch_num / num_batches_to_evaluate) * self.config.SAVE_EVERY_EPOCHS)
                        save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                        self.save_model(self.sess, save_target)
                        print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                        results, precision, recall, f1 = self.evaluate()
                        print('Accuracy after %d epochs: %s' % (epoch_num, results[:5]))
                        print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                            recall) + ', F1: ' + str(f1))

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def guard_code_batch(self, batch, word_embeddings, threshold=0):

        # vunk
        # result = [guard_by_vunk(r) for r in batch]

        # do nothing
        # result = [common_adversarial.separate_vars_code(r)[1] for r in batch]

        # distance
        result = [codeguard.guard_by_distance(r, lambda w: w in self.word_to_index,
                                              lambda w: word_embeddings[self.word_to_index[w]], threshold)
                  for r in batch]
        return result


    def create_ordered_words_dictionary(self, data_path, top_words):
        with open('{}.dict.c2v'.format(data_path), 'rb') as file:
            word_to_count = pickle.load(file)

        all_words = list(word_to_count.keys())

        # filter reserved words and invalid variable names
        reserved_java_words =["abstract", "assert", "boolean", "break", "byte", "case",
                              "catch", "char", "class", "const", "continue", "default",
                              "double", "do", "else", "enum", "extends", "false",
                              "final", "finally", "float", "for", "goto", "if",
                              "implements", "import", "instanceof", "int", "interface", "long",
                              "native", "new", "null", "package", "private", "protected",
                              "public", "return", "short", "static", "strictfp", "super",
                              "switch", "synchronized", "this", "throw", "throws", "transient",
                              "true", "try", "void", "volatile", "while"]
        valid_varname_matcher = re.compile("^[a-z_][a-z_0-9]*$")

        all_words = [w for w in all_words if valid_varname_matcher.match(w)
                     and w not in reserved_java_words]

        all_words.sort(key=lambda w: (-word_to_count[w], w))

        indextop_to_word = all_words[:top_words]
        word_to_indextop = {w: i for i, w in enumerate(indextop_to_word)}
        print('Dictionaries ordered-by count created. (created from: {}.dict.c2v)'.format(data_path))

        return word_to_indextop, indextop_to_word

    def evaluate_and_adverse(self, depth, topk, targeted_attack, adversarial_target_word,
                             deadcode_attack, guard_input = None, adverse_TP_only = True, data_dict_path = None):

        topk_words_from_model = 100
        data_dict_path = data_dict_path if data_dict_path is not None else \
            Model.get_data_dictionaries_path(self.config.LOAD_PATH)
        word_to_indextop, indextop_to_word  = self.create_ordered_words_dictionary(
            data_dict_path,
            self.config.MAX_WORDS_FROM_VOCAB_FOR_ADVERSARIAL)

        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                  path_to_index=self.path_to_index,
                                                                  target_word_to_index=self.target_word_to_index,
                                                                  config=self.config, is_evaluating=True)
            self.eval_placeholder = self.eval_queue.get_input_placeholder()
            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, \
                    self.eval_source_string, _, self.eval_path_target_string= \
                self.build_test_graph(self.eval_queue.get_filtered_batches())
            self.saver = tf.train.Saver()

        if self.grad_wrt_input is None:
            self.words_to_compute_grads = tf.placeholder(tf.string, [None])

            self.loss_wrt_input, self.grad_wrt_input, self.adversarial_name, self.adversarial_name_index = \
                self.build_test_graph_with_loss(self.eval_queue.get_filtered_batches(), self.eval_queue,
                                                indextop_to_word, self.words_to_compute_grads, topk_words_from_model)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name )
                self.saver.save(self.sess, release_name )
                return None

        if self.eval_data_lines is None:
            print('Loading test data from: ' + self.config.TEST_PATH)
            self.eval_data_lines = common.load_file_lines(self.config.TEST_PATH)
            print('Done loading test data')

        with open('log.txt', 'w') as output_file:
            total_fools = 0
            total_failed = 0
            results = []

            # for deadcode
            if deadcode_attack:
                variable_picker = adversarialsearcher.init_deadcode_variable
            else:
                variable_picker = None

            # untargeted searcher
            if not targeted_attack:
                print("Using non-targeted attack")
                all_searchers = [AdversarialSearcher(topk, depth, word_to_indextop, indextop_to_word,
                                                     line, variable_picker, random_start=True)
                                 for line in self.eval_data_lines]
            else: # targeted searcher
                if adversarial_target_word == "random-uniform":
                    print("Using targeted attack. target sampled uniform-ly")
                    get_name = lambda : random.choice(list(self.target_word_to_index.keys()))
                elif adversarial_target_word == "random-unigram":
                    print("Using targeted attack. target sampled unigram-ly")
                    with open(self.get_target_words_histogram_path(self.config.LOAD_PATH), 'rb') as file:
                        histogram = pickle.load(file)

                    words, weight = list(zip(*histogram.items()))

                    get_name = lambda: random.choices(words, weight)[0]
                else:
                    print("Using targeted attack. target:", adversarial_target_word)
                    if adversarial_target_word not in self.target_word_to_index:
                        print(adversarial_target_word, "not existed in vocab!")
                        return []
                    get_name = lambda: adversarial_target_word
                all_searchers = [AdversarialTargetedSearcher(topk, depth, word_to_indextop, indextop_to_word,
                                                             line, get_name(), variable_picker, random_start=True)
                                 for line in self.eval_data_lines]

            all_searchers = [[None, se] for se in all_searchers if se.can_be_adversarial()]

            del self.eval_data_lines
            self.eval_data_lines = None

            if guard_input is not None:
                word_embeddings = self.get_words_vocab_embed()
                print("Guard input is active. (make sure dataset includes variables-list)")
            print("Total adversariable data:", len(all_searchers))
            print("Proccesing in batches of:", self.config.TEST_BATCH_SIZE,
                  "adversarial mini-batches:", self.config.ADVERSARIAL_MINI_BATCH_SIZE)
            i=0
            processed = 0
            excluded = 0
            trivial = 0
            batch_searchers = []
            # print("\ttrue_name\ttrue_prediction\tadversarial_prediction\tstate")
            output_file.write("\ttrue_name\ttrue_prediction\tadversarial_prediction\tstate\n")
            while all_searchers or batch_searchers :
                if len(batch_searchers) < self.config.TEST_BATCH_SIZE:
                    free_slots = self.config.TEST_BATCH_SIZE - len(batch_searchers)
                    processed += len(all_searchers[:free_slots])
                    new_batch = all_searchers[:free_slots]
                    del all_searchers[:free_slots]
                    batch_searchers += new_batch

                if guard_input is not None:
                    batch_nodes_data = [(se, n, c) for se in batch_searchers
                                        for n, c in se[1].pop_unchecked_adversarial_code(return_with_vars=True)]

                    batch_nodes_data = [(se, n, codeguard.guard_by_distance(c, lambda w: w in self.word_to_index,
                                                                        lambda w: word_embeddings[self.word_to_index[w]],
                                                                            guard_input))
                                        for se, n, c in batch_nodes_data]
                else:
                    batch_nodes_data = [(se, n, c) for se in batch_searchers
                                        for n, c in se[1].pop_unchecked_adversarial_code()]
                batch_data = [c for _, _, c in batch_nodes_data]

                top_words, top_scores, original_names = self.sess.run(
                    [self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op],
                    feed_dict={self.eval_placeholder: batch_data})
                top_words = common.binary_to_string_matrix(top_words)

                new_batch_searchers = []
                searcher_done = {}
                for (searcher, node, _), one_top_words in zip(batch_nodes_data, top_words):
                    # if already found - skip
                    if searcher[1] in searcher_done:
                        continue

                    one_top_words = common.filter_impossible_names(one_top_words)
                    if not one_top_words:
                        output_file.write("code with state: " +
                                          str(node) + " cause empty predictions\n")
                        continue

                    # save original prediction
                    if searcher[0] is None:
                        searcher[0] = one_top_words[0]
                        # filter wrong examples (examples that originally predicted wrong)
                        if adverse_TP_only and searcher[0] != searcher[1].get_original_name():
                            excluded += 1
                            continue

                    if searcher[1].is_target_found(one_top_words):
                        if (not targeted_attack and searcher[0] == searcher[1].get_original_name()) or \
                                (targeted_attack and searcher[1].get_adversarial_name() != searcher[1].get_original_name()):
                            total_fools += 1
                        else:
                            trivial += 1

                        searcher_done[searcher[1]] = None

                        out = "\t" + searcher[1].get_original_name() +\
                              "\t" + searcher[0] +\
                              "\t" + one_top_words[0] + \
                              "\t" + str(node)

                        output_file.write(out + "\n")
                        continue

                    if searcher not in new_batch_searchers:
                        new_batch_searchers.append(searcher)

                batch_searchers = [s for s in new_batch_searchers if s[1] not in searcher_done]
                batch_data = [se[1].get_adversarial_code() for se in batch_searchers]
                batch_word_to_derive = [se[1].get_word_to_derive() for se in batch_searchers]
                # if all methods fails - continue without grad calculation
                if not batch_searchers:
                    continue

                new_batch_searchers = []

                while batch_data:
                    mini_batch_searchers = batch_searchers[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]
                    mini_batch_data = batch_data[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]
                    mini_batch_word_to_derive = batch_word_to_derive[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]
                    del batch_searchers[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]
                    del batch_data[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]
                    del batch_word_to_derive[:self.config.ADVERSARIAL_MINI_BATCH_SIZE]

                    loss_of_input, grad_of_input  = self.sess.run(
                        [self.loss_wrt_input,
                         self.grad_wrt_input],
                        feed_dict={self.eval_placeholder: mini_batch_data, self.words_to_compute_grads: mini_batch_word_to_derive})

                    # source_target_strings = np.array(common.binary_to_string_matrix(source_target_strings))

                    for searcher, grads in zip(mini_batch_searchers, grad_of_input):
                        if not searcher[1].next((0, "", grads)):
                            total_failed += 1
                            out = "\t" + searcher[1].get_original_name() + \
                                  "\t" + searcher[0] + \
                                  "\t" + "--FAIL--" + \
                                  "\t" + str(searcher[1].get_current_node())
                            output_file.write(out + "\n")
                            continue

                        new_batch_searchers.append(searcher)

                batch_searchers = new_batch_searchers

                if i % self.num_batches_to_log == 0:
                    print("batch:", i, "processed:", processed, "(excluded:", excluded, "trivial:", trivial, ")",
                          "fools: " + str(total_fools) + " fail to fool: " + str(total_failed) +
                          " success rate: {:.2f}%".format(100 * total_fools / (total_fools+total_failed)
                                                                            if total_fools+total_failed > 0 else 0))
                i += 1

            print('Done testing, epoch reached')

            print("Final Results:", "processed:", processed, "(excluded:", excluded, "trivial:", trivial, ")",
                          "fools: " + str(total_fools) + " fail to fool: " + str(total_failed) +
                          " success rate: {:.2f}%".format(100 * total_fools / (total_fools+total_failed)
                                                                            if total_fools+total_failed > 0 else 0))
            output_file.write("processed: " + str(processed) + " (excluded:" + str(excluded) + " trivial:" + str(trivial) + ")" +
                              " fools: " + str(total_fools) + " fail to fool: " + str(total_failed) +
                              " success rate: {:.2f}%".format(100 * total_fools / (total_fools+total_failed)
                                                                            if total_fools+total_failed > 0 else 0) + '\n')

        elapsed = int(time.time() - eval_start_time)
        print("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return results

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, top_words in results:
            prediction = common.filter_impossible_names(top_words)[0]
            original_subtokens = common.get_subtokens(original_name)
            predicted_subtokens = common.get_subtokens(prediction)
            for subtok in predicted_subtokens:
                if subtok in original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in original_subtokens:
                if not subtok in predicted_subtokens:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed, total_examples):
        state_message = 'Evaluated %d/%d examples...' % (total_predictions, total_examples)
        throughput_message = "Prediction throughput: %d samples/sec" % int(total_predictions / (elapsed if elapsed > 0 else 1))
        print(state_message)
        print(throughput_message)

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, top_words in results:
            normalized_original_name = common.normalize_word(original_name)
            predicted_something = False
            for i, predicted_word in enumerate(common.filter_impossible_names(top_words)):
                if i == 0:
                    output_file.write('Original: ' + original_name + ', predicted 1st: ' + predicted_word + '\n')
                predicted_something = True
                normalized_suggestion = common.normalize_word(predicted_word)
                if normalized_original_name == normalized_suggestion:
                    output_file.write('\t\t predicted correctly at rank: ' + str(i + 1) + '\n')
                    for j in range(i, self.topk):
                        num_correct_predictions[j] += 1
                    break
            if not predicted_something:
                output_file.write('No results for predicting: ' + original_name)
        return num_correct_predictions

    def build_training_graph(self, input_tensors):
        words_input, source_input, path_input, target_input, valid_mask = input_tensors  # (batch, 1),   (batch, max_contexts)

        with tf.variable_scope('model'):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1), dtype=tf.float32)
            paths_vocab = tf.get_variable('PATHS_VOCAB', shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))

            weighted_average_contexts, _ = self.calculate_weighted_contexts(words_vocab, paths_vocab, attention_param,
                                                                            source_input, path_input, target_input,
                                                                            valid_mask)

            logits = tf.matmul(weighted_average_contexts, target_words_vocab, transpose_b=True)
            batch_size = tf.to_float(tf.shape(words_input)[0])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(words_input, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def build_adversarial_training_graph_with_loss(self, input_tensors, queue,
                                                        adversary_words_in_vocab, words_to_compute_grads,
                                                        topk_results = None):
        words_to_compute_grads = tf.reshape(words_to_compute_grads, [-1, 1])
        # with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('model'):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1), dtype=tf.float32)
            paths_vocab = tf.get_variable('PATHS_VOCAB', shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))

            # target_words_vocab = tf.transpose(target_words_vocab)  # (dim, word_vocab+1)

            words_input, source_input, path_input, target_input, valid_mask, source_string, path_string, path_target_string = input_tensors  # (batch, 1), (batch, max_contexts)

            words_input_index = queue.target_word_table.lookup(words_input)

            ### PART 1: build training computation graph
            weighted_average_contexts, _ = self.calculate_weighted_contexts(words_vocab, paths_vocab, attention_param,
                                                                            source_input, path_input, target_input,
                                                                            valid_mask)
            logits = tf.matmul(weighted_average_contexts, target_words_vocab, transpose_b=True)

            batch_size = tf.to_float(tf.shape(words_input)[0])
            train_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(words_input_index, [-1]),
                logits=logits)) / batch_size

            train_optimizer = tf.train.AdamOptimizer().minimize(train_loss)

            ### PART 2: build adversarial computation graph (same graph as the graph for adversary (for grads computation))
            weighted_average_contexts, attention_weights, source_word_embed, target_word_embed = \
                self.calculate_weighted_contexts(words_vocab, paths_vocab,
                        attention_param,
                        source_input, path_input,
                        target_input,
                        valid_mask, True, return_embed=True)

            logits = tf.matmul(weighted_average_contexts, target_words_vocab, transpose_b=True)

            original_words = words_input
            original_words_index = words_input_index
            adversary_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(words_input_index, [-1]),
                logits=logits)

            grad = tf.gradients([adversary_loss], [source_word_embed, target_word_embed])
            grad_source_word_embed, grad_target_word_embed = grad
            grad_word_embed = tf.concat([grad_source_word_embed, grad_target_word_embed], axis=1)

            # filter unrelevant var names
            path_source_target_tensor = tf.cond(
                tf.not_equal(tf.rank(source_string), 1),
                lambda: tf.concat([source_string, path_target_string], axis=1),
                lambda: tf.concat(
                    [tf.expand_dims(source_string, axis=0),
                     tf.expand_dims(path_target_string, axis=0)],
                    axis=1))

            # create vocab for adversarial (by given words)
            partial_words_vocab = tf.gather(words_vocab, [self.word_to_index[w] for w in adversary_words_in_vocab])

            # filter only given vars
            mask = tf.equal(words_to_compute_grads, path_source_target_tensor)
            grad_word_embed = tf.reduce_sum(tf.multiply(grad_word_embed, tf.expand_dims(tf.cast(mask, tf.float32), -1)),
                                            axis=1)

            grad_of_input = tf.matmul(grad_word_embed, partial_words_vocab, transpose_b=True)

            if topk_results is None:
                batched_grad_of_source_input = grad_of_input
            else:
                top_values, top_indices  = tf.math.top_k(grad_of_input, k=topk_results)
                bottom_values, bottom_indices = tf.math.top_k(-grad_of_input, k=topk_results)
                values = tf.concat([top_values, -bottom_values], axis=-1)
                indices = tf.concat([top_indices, bottom_indices], axis=-1)
                batched_grad_of_source_input = tf.stack([tf.cast(indices, tf.float32), values], axis=1)

        return adversary_loss, batched_grad_of_source_input, original_words, original_words_index, \
               train_optimizer, train_loss

    def calculate_weighted_contexts(self, words_vocab, paths_vocab, attention_param, source_input, path_input,
                                    target_input, valid_mask, is_evaluating=False, return_embed = False):
        keep_prob1 = 0.75
        max_contexts = self.config.MAX_CONTEXTS

        source_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=source_input)  # (batch, max_contexts, dim)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=path_input)  # (batch, max_contexts, dim)
        target_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=target_input)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
                                  axis=-1)  # (batch, max_contexts, dim * 3)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, keep_prob1)

        flat_embed = tf.reshape(context_embed, [-1, self.config.EMBEDDINGS_SIZE * 3])  # (batch * max_contexts, dim * 3)
        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.EMBEDDINGS_SIZE * 3, self.config.EMBEDDINGS_SIZE * 3),
                                          dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

        contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
        batched_contexts_weights = tf.reshape(contexts_weights,
                                              [-1, max_contexts, 1])  # (batch, max_contexts, 1)
        mask = tf.log(valid_mask)  # (batch, max_contexts)
        mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
        batched_contexts_weights += mask  # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, dim=1)  # (batch, max_contexts, 1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, max_contexts, self.config.EMBEDDINGS_SIZE * 3])
        weighted_average_contexts = tf.reduce_sum(tf.multiply(batched_embed, attention_weights),
                                                  axis=1)  # (batch, dim * 3)
        if not return_embed:
            return weighted_average_contexts, attention_weights
        else:
            return weighted_average_contexts, attention_weights, source_word_embed, target_word_embed

    def build_test_graph(self, input_tensors, normalize_scores=False):
        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            self.words_vocab_embed = words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32, trainable=False)
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1),
                                              dtype=tf.float32, trainable=False)
            paths_vocab = tf.get_variable('PATHS_VOCAB',
                                          shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            target_words_vocab = tf.transpose(target_words_vocab)  # (dim, word_vocab+1)

            words_input, source_input, path_input, target_input, valid_mask, source_string, path_string, path_target_string = input_tensors  # (batch, 1), (batch, max_contexts)

            weighted_average_contexts, attention_weights = self.calculate_weighted_contexts(words_vocab, paths_vocab,
                                                                                            attention_param,
                                                                                            source_input, path_input,
                                                                                            target_input,
                                                                                            valid_mask, True)

        cos = tf.matmul(weighted_average_contexts, target_words_vocab)

        topk_candidates = tf.nn.top_k(cos, k=tf.minimum(self.topk, self.target_word_vocab_size))
        top_indices = tf.to_int64(topk_candidates.indices)
        top_words = self.index_to_target_word_table.lookup(top_indices)
        original_words = words_input
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, attention_weights, source_string, path_string, path_target_string

    def build_test_graph_with_loss(self, input_tensors, queue, adversary_words_in_vocab, words_to_compute_grads,
                                   topk_results = None):
        words_to_compute_grads = tf.reshape(words_to_compute_grads, [-1, 1])
        with tf.variable_scope('model', reuse=True):

            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32, trainable=False)
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1),
                                              dtype=tf.float32, trainable=False)
            paths_vocab = tf.get_variable('PATHS_VOCAB',
                                          shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            target_words_vocab = tf.transpose(target_words_vocab)  # (dim, word_vocab+1)

            words_input, source_input, path_input, target_input, valid_mask, source_string, path_string, path_target_string = input_tensors  # (batch, 1), (batch, max_contexts)

            weighted_average_contexts, attention_weights, source_word_embed, target_word_embed = \
                self.calculate_weighted_contexts(words_vocab, paths_vocab,
                        attention_param,
                        source_input, path_input,
                        target_input,
                        valid_mask, True, return_embed=True)

            logits = cos = tf.matmul(weighted_average_contexts, target_words_vocab)

            batch_size = tf.to_float(tf.shape(words_input)[0])
            words_input_index = queue.target_word_table.lookup(words_input)

            original_words = words_input
            original_words_index = words_input_index
            # loss = tf.reduce_sum(
            loss =    tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(words_input_index, [-1]),
                logits=logits)

            grad = tf.gradients([loss], [source_word_embed, target_word_embed])
            grad_source_word_embed, grad_target_word_embed = grad
            grad_word_embed = tf.concat([grad_source_word_embed, grad_target_word_embed], axis=1)

            # filter unrelevant var names
            path_source_target_tensor = tf.cond(
                tf.not_equal(tf.rank(source_string), 1),
                lambda: tf.concat([source_string, path_target_string], axis=1),
                lambda: tf.concat(
                    [tf.expand_dims(source_string, axis=0),
                     tf.expand_dims(path_target_string, axis=0)],
                    axis=1))

            # create vocab for adversarial (by given words)
            partial_words_vocab = tf.gather(words_vocab, [self.word_to_index[w] for w in adversary_words_in_vocab])

            # filter only given vars
            mask = tf.equal(words_to_compute_grads, path_source_target_tensor)
            grad_word_embed = tf.reduce_sum(tf.multiply(grad_word_embed, tf.expand_dims(tf.cast(mask, tf.float32), -1)),
                                            axis=1)

            grad_of_input = tf.matmul(grad_word_embed, partial_words_vocab, transpose_b=True)

            if topk_results is None:
                batched_grad_of_source_input = grad_of_input
            else:
                top_values, top_indices  = tf.math.top_k(grad_of_input, k=topk_results)
                bottom_values, bottom_indices = tf.math.top_k(-grad_of_input, k=topk_results)
                values = tf.concat([top_values, -bottom_values], axis=-1)
                indices = tf.concat([top_indices, bottom_indices], axis=-1)
                batched_grad_of_source_input = tf.stack([tf.cast(indices, tf.float32), values], axis=1)

        return loss, batched_grad_of_source_input, original_words, original_words_index

    def predict(self, predict_data_lines, guard_input=None):
        if self.predict_queue is None:
            self.predict_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                     path_to_index=self.path_to_index,
                                                                     target_word_to_index=self.target_word_to_index,
                                                                     config=self.config, is_evaluating=True)
            self.predict_placeholder = self.predict_queue.get_input_placeholder()
            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
            self.attention_weights_op, self.predict_source_string, self.predict_path_string, self.predict_path_target_string = \
                self.build_test_graph(self.predict_queue.get_filtered_batches(), normalize_scores=True)

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        if guard_input is not None:
            word_embeddings = self.get_words_vocab_embed()
            print("Guard input is active. (make sure dataset includes variables-list)")

        results = []
        for batch in common.split_to_batches(predict_data_lines, 1):
            if guard_input is not None:
                batch = self.guard_code_batch(batch, word_embeddings, guard_input)

            top_words, top_scores, original_names, attention_weights, source_strings, path_strings, target_strings = self.sess.run(
                [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op,
                 self.attention_weights_op, self.predict_source_string, self.predict_path_string,
                 self.predict_path_target_string],
                feed_dict={self.predict_placeholder: batch})
            top_words, original_names = common.binary_to_string_matrix(top_words), common.binary_to_string_matrix(
                original_names)
            # Flatten original names from [[]] to []
            attention_per_path = self.get_attention_per_path(source_strings, path_strings, target_strings,
                                                             attention_weights)
            original_names = [w for l in original_names for w in l]
            results.append((original_names[0], top_words[0], top_scores[0], attention_per_path))
        return results

    def calc_loss_and_gradients_wrt_input(self, predict_data_lines, word_to_derive, adversary_words_in_vocab):

        topk_words_from_model = 100
        if self.predict_queue is None:
            self.predict_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                     path_to_index=self.path_to_index,
                                                                     target_word_to_index=self.target_word_to_index,
                                                                     config=self.config, is_evaluating=True)
            self.predict_placeholder = self.predict_queue.get_input_placeholder()
            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
            self.attention_weights_op, self.predict_source_string, self.predict_path_string, self.predict_path_target_string = \
                self.build_test_graph(self.predict_queue.get_filtered_batches(), normalize_scores=True)

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        if self.grad_wrt_input is None:
            self.words_to_compute_grads = tf.placeholder(tf.string, [None])

            self.loss_wrt_input, self.grad_wrt_input, self.adversarial_name, self.adversarial_name_index = \
                self.build_test_graph_with_loss(self.predict_queue.get_filtered_batches(), self.predict_queue,
                                                adversary_words_in_vocab, self.words_to_compute_grads, topk_words_from_model)

        # for batch in common.split_to_batches(predict_data_lines, 1):
        batch = predict_data_lines
        loss_of_input, grad_of_input = self.sess.run(
            [self.loss_wrt_input,
             self.grad_wrt_input],
            feed_dict={self.predict_placeholder: batch, self.words_to_compute_grads: word_to_derive})

        return loss_of_input, grad_of_input

    def get_words_vocab_embed(self, word = None):
        if word is None:
            result_words_vocab_embed = self.sess.run(self.words_vocab_embed)
            return result_words_vocab_embed
        else:
            if type(word) == list:
                embed_word = [self.words_vocab_embed[self.word_to_index[w]] for w in word]
                result_words_vocab_embed = self.sess.run(embed_word)
                return result_words_vocab_embed
            else:
                if word in self.word_to_index:
                    result_words_vocab_embed = self.sess.run(self.words_vocab_embed[self.word_to_index[word]])
                    return result_words_vocab_embed
                else:
                    return None

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        attention_weights = np.squeeze(attention_weights)  # (max_contexts, )
        attention_per_context = {}
        for source, path, target, weight in zip(source_strings, path_strings, target_strings, attention_weights):
            string_triplet = (
                common.binary_to_string(source), common.binary_to_string(path), common.binary_to_string(target))
            attention_per_context[string_triplet] = weight
        return attention_per_context

    @staticmethod
    def get_dictionaries_path(model_file_path):
        dictionaries_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])

    @staticmethod
    def get_target_words_histogram_path(model_file_path):
        dictionaries_save_file_name = "target_words_histogram.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])

    @staticmethod
    def get_data_dictionaries_path(model_file_path):
        data_dictionaries_save_file_name = model_file_path.split('/')[-2]
        return '/'.join(model_file_path.split('/')[:-1] + [data_dictionaries_save_file_name])

    def save_model(self, sess, path):
        self.saver.save(sess, path)
        with open(self.get_dictionaries_path(path), 'wb') as file:
            pickle.dump(self.word_to_index, file)
            pickle.dump(self.index_to_word, file)
            pickle.dump(self.word_vocab_size, file)

            pickle.dump(self.target_word_to_index, file)
            pickle.dump(self.index_to_target_word, file)
            pickle.dump(self.target_word_vocab_size, file)

            pickle.dump(self.path_to_index, file)
            pickle.dump(self.index_to_path, file)
            pickle.dump(self.path_vocab_size, file)

    def load_model(self, sess):
        if not sess is None:
            print('Loading model weights from: ' + self.config.LOAD_PATH)
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done')
        dictionaries_path = self.get_dictionaries_path(self.config.LOAD_PATH)
        with open(dictionaries_path , 'rb') as file:
            print('Loading model dictionaries from: %s' % dictionaries_path)
            self.word_to_index = pickle.load(file)
            self.index_to_word = pickle.load(file)
            self.word_vocab_size = pickle.load(file)

            self.target_word_to_index = pickle.load(file)
            self.index_to_target_word = pickle.load(file)
            self.target_word_vocab_size = pickle.load(file)

            self.path_to_index = pickle.load(file)
            self.index_to_path = pickle.load(file)
            self.path_vocab_size = pickle.load(file)
            print('Done')

    def save_word2vec_format(self, dest, source):
        with tf.variable_scope('model'):
            if source is VocabType.Token:
                vocab_size = self.word_vocab_size
                embedding_size = self.config.EMBEDDINGS_SIZE
                index = self.index_to_word
                var_name = 'WORDS_VOCAB'
            elif source is VocabType.Target:
                vocab_size = self.target_word_vocab_size
                embedding_size = self.config.EMBEDDINGS_SIZE * 3
                index = self.index_to_target_word
                var_name = 'TARGET_WORDS_VOCAB'
            else:
                raise ValueError('vocab type should be VocabType.Token or VocabType.Target.')
            embeddings = tf.get_variable(var_name, shape=(vocab_size + 1, embedding_size), dtype=tf.float32,
                                     trainable=False)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)
            np_embeddings = self.sess.run(embeddings)
        with open(dest, 'w') as words_file:
            common.save_word2vec_file(words_file, vocab_size, embedding_size, index, np_embeddings)

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
