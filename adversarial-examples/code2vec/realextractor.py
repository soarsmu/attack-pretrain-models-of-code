import subprocess
import numpy as np
import preprocess_test_batch
from extractor import Extractor


class RealExtractor(Extractor):
    def __init__(self, config, jar_path, max_path_length, max_path_width, path_dict_and_name):
        Extractor.__init__(self,config,jar_path,max_path_length,max_path_width)
        self.word_to_count, self.path_to_count, _, _ = preprocess_test_batch.load_dictionaries(path_dict_and_name)

    def select_sample_or_all_contexts(self, select_from_contexts, max_contexts):
        if len(select_from_contexts) <= max_contexts:
            return select_from_contexts

        return preprocess_test_batch.pick_contexts(select_from_contexts, max_contexts, self.path_to_count,
                                                   self.word_to_count,
                                                   RealExtractor.context_partial_found_by_full_path,
                                                   RealExtractor.context_full_found_by_full_path)

    def extract_paths(self, path):
        command = ['java', '-cp', self.jar_path, 'JavaExtractor.App', '--max_path_length',
                   str(self.max_path_length), '--max_path_width', str(self.max_path_width), '--file', path, '--no_hash']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        output = out.decode().splitlines()
        if len(output) == 0:
            err = err.decode()
            raise ValueError(err)
        hash_to_string_dict = {}
        result = []
        for i, line in enumerate(output):
            parts = line.rstrip().split(' ')
            method_name = parts[0]
            current_result_line_parts = [method_name]
            contexts = parts[1:]

            for context in self.select_sample_or_all_contexts(contexts,self.config.MAX_CONTEXTS):
                context_parts = context.split(',')
                context_word1 = context_parts[0]
                context_path = context_parts[1]
                context_word2 = context_parts[2]
                hashed_path = str(self.java_string_hashcode(context_path))
                hash_to_string_dict[hashed_path] = context_path
                current_result_line_parts += ['%s,%s,%s' % (context_word1, hashed_path, context_word2)]
            space_padding = ' ' * (self.config.MAX_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result, hash_to_string_dict

    @staticmethod
    def java_string_hashcode(s):
        """
        Imitating Java's String#hashCode, because the model is trained on hashed paths but we wish to
        Present the path attention on un-hashed paths.
        """
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000

    @staticmethod
    def context_full_found_by_full_path(context_parts, word_to_count, path_to_count):
        return context_parts[0] in word_to_count \
               and str(RealExtractor.java_string_hashcode(context_parts[1])) in path_to_count \
               and context_parts[2] in word_to_count

    @staticmethod
    def context_partial_found_by_full_path(context_parts, word_to_count, path_to_count):
        return context_parts[0] in word_to_count \
               or str(RealExtractor.java_string_hashcode(context_parts[1])) in path_to_count \
               or context_parts[2] in word_to_count


class RealExtractorForAdversarial(RealExtractor):
    def __init__(self, config, jar_path, max_path_length, max_path_width, path_dict_and_name):
        super().__init__(config,jar_path,max_path_length,max_path_width, path_dict_and_name)

    def extract_paths(self, path):
        command = ['java', '-cp', self.jar_path, 'JavaExtractor.App', '--max_path_length',
                   str(self.max_path_length), '--max_path_width', str(self.max_path_width), '--file', path, '--no_hash']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        output = out.decode().splitlines()
        if len(output) == 0:
            err = err.decode()
            raise ValueError(err)
        hash_to_string_dict = {}
        result = []
        for i, line in enumerate(output):
            parts = line.rstrip().split(' ')

            vars = parts[0]
            method_name = parts[1]
            current_result_line_parts = [vars, method_name]
            contexts = parts[2:]

            for context in self.select_sample_or_all_contexts(contexts,self.config.MAX_CONTEXTS):
                context_parts = context.split(',')
                context_word1 = context_parts[0]
                context_path = context_parts[1]
                context_word2 = context_parts[2]
                hashed_path = str(self.java_string_hashcode(context_path))
                hash_to_string_dict[hashed_path] = context_path
                current_result_line_parts += ['%s,%s,%s' % (context_word1, hashed_path, context_word2)]
            space_padding = ' ' * (self.config.MAX_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result, hash_to_string_dict