import numpy as np
import random

START_ADVERSARY_ALPHABET = 2
END_ADVERSARY_ALPHABET = 28

def get_label_length(unique_label_as_ints):
    if any(unique_label_as_ints == 0):
        return np.argmax(unique_label_as_ints == 0)
    return unique_label_as_ints.size

def construct_name_from_ints(unique_label_as_ints, index_to_alphabet):
    length = get_label_length(unique_label_as_ints)
    return "".join([index_to_alphabet(unique_label_as_ints[i]) for i in range(length)])


def adversary_by_prefix_random(unique_label_to_adverse_as_ints, prefix_count):
    """

    :param unique_label_to_adverse_as_ints:
    :param unique_label_to_adverse_grads:
    :param prefix_count: how much characters to replace. if -1 then all characters will be replaced.
    :return:
    """
    if prefix_count == -1:
        prefix_count = get_label_length(unique_label_to_adverse_as_ints)

    for i in range(prefix_count):
        target_char = random.randint(START_ADVERSARY_ALPHABET,END_ADVERSARY_ALPHABET)
        unique_label_to_adverse_as_ints[i] = target_char

    return unique_label_to_adverse_as_ints


def adversary_by_prefix_rename(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads, prefix_count):
    """
    :param unique_label_to_adverse_as_ints:
    :param unique_label_to_adverse_grads:
    :param prefix_count: how much characters to replace. if -1 then all characters will be replaced.
    :return:
    """
    if prefix_count == -1:
        prefix_count = get_label_length(unique_label_to_adverse_as_ints)

    for i in range(prefix_count):
        target_char = np.argmax(unique_label_to_adverse_grads[i]) + START_ADVERSARY_ALPHABET
        unique_label_to_adverse_as_ints[i] = target_char

    return unique_label_to_adverse_as_ints

def adversary_by_argmax_id(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads):
    unique_label_to_adverse_length = get_label_length(unique_label_to_adverse_as_ints)

    target_index = np.argmax(np.max(unique_label_to_adverse_grads, axis=1))
    target_char = np.argmax(unique_label_to_adverse_grads[target_index]) + START_ADVERSARY_ALPHABET
    unique_label_to_adverse_as_ints[target_index] = target_char

    for i in range(unique_label_to_adverse_length, target_index):
        unique_label_to_adverse_as_ints[i] = np.argmax(unique_label_to_adverse_grads[i]) + START_ADVERSARY_ALPHABET

    return unique_label_to_adverse_as_ints

def adversary_until_argmax_id(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads):
    unique_label_to_adverse_length = get_label_length(unique_label_to_adverse_as_ints)

    target_index = np.argmax(np.max(unique_label_to_adverse_grads, axis=1))
    target_char = np.argmax(unique_label_to_adverse_grads[target_index]) + START_ADVERSARY_ALPHABET
    unique_label_to_adverse_as_ints[target_index] = target_char

    for i in range(target_index):
        unique_label_to_adverse_as_ints[i] = np.argmax(unique_label_to_adverse_grads[i]) + START_ADVERSARY_ALPHABET

    return unique_label_to_adverse_as_ints

def adversary_all_or_until_argmax_id(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads):
    unique_label_to_adverse_length = get_label_length(unique_label_to_adverse_as_ints)

    target_index = np.argmax(np.max(unique_label_to_adverse_grads, axis=1))
    target_char = np.argmax(unique_label_to_adverse_grads[target_index]) + START_ADVERSARY_ALPHABET
    unique_label_to_adverse_as_ints[target_index] = target_char

    for i in range(max(target_index, unique_label_to_adverse_length)):
        unique_label_to_adverse_as_ints[i] = np.argmax(unique_label_to_adverse_grads[i]) + START_ADVERSARY_ALPHABET

    return unique_label_to_adverse_as_ints

def adversary_all_or_until_top_and_index(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads,
                                      index_place, char_place):
    unique_label_to_adverse_length = get_label_length(unique_label_to_adverse_as_ints)


    target_index = np.argsort(np.sort(unique_label_to_adverse_grads, axis=-1)[:, -char_place])[-index_place]
    target_chars = np.argsort(unique_label_to_adverse_grads, axis=-1)[:, -char_place] + START_ADVERSARY_ALPHABET

    copy_until = max(target_index+1, unique_label_to_adverse_length)

    np.copyto(unique_label_to_adverse_as_ints[:copy_until], target_chars.astype('uint8')[:copy_until])

    return unique_label_to_adverse_as_ints

def adversary_all19_by_argmax(unique_label_to_adverse_as_ints, unique_label_to_adverse_grads):

    target_chars = np.argmax(unique_label_to_adverse_grads, axis=-1) + START_ADVERSARY_ALPHABET

    np.copyto(unique_label_to_adverse_as_ints, target_chars.astype('uint8'))

    return unique_label_to_adverse_as_ints

def adversary_all19_by_argmax_batch(unique_label_to_adverse_grads):

    target_chars = np.argmax(unique_label_to_adverse_grads, axis=-1) + START_ADVERSARY_ALPHABET

    return target_chars.astype('uint8')