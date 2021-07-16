import multiprocessing as mp
import os
import sys
import preprocess_test_batch

CONTINUE_PROCESS=True
PYTHON="python3"
EXTRACTOR="JavaExtractor/extract.py"
NUM_THREADS=1
REMOVE="rm"
# REMOVE="del"

dictionaries_path = "data/java-large/java-large"
word_to_count, path_to_count, target_to_count, num_training_examples = \
        preprocess_test_batch.load_dictionaries(dictionaries_path)

def preprocess_dir(path:str):

    for f in os.listdir(path + "/src"):
        f_name = f[:-5]
        if (not CONTINUE_PROCESS) or (not os.path.exists(os.path.join(path, f_name + ".test.c2v"))):
            f_absolutepath = os.path.join(path, "src", f)
            #extract paths
            os.system(
                PYTHON + " " + EXTRACTOR + " --file " + f_absolutepath + " --max_path_length 8 --max_path_width 2 --num_threads " + str(
                    NUM_THREADS) +
                " --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > " + path + "/" + f_name + "tempmut")

            #preprocess paths
            test_data_path = path + "/" + f_name + "tempmut"
            output_name = path + "/" + f_name
            max_contexts = 200
            for data_file_path, data_role in zip([test_data_path], ['test']):
                num_examples = preprocess_test_batch.process_file(file_path=data_file_path, data_file_role=data_role,
                                            dataset_name=output_name,
                                            word_to_count=word_to_count, path_to_count=path_to_count,
                                            max_contexts=int(max_contexts))

            os.remove(path + "/" + f_name + "tempmut")

if __name__ == "__main__":

    mypath = sys.argv[1]
    alldirs = [os.path.join(mypath, f).replace("\\", "/") for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]

    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(preprocess_dir, alldirs)