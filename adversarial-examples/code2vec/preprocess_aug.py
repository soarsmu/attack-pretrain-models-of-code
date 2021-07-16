import multiprocessing as mp
import os
import sys

CONTINUE_PROCESS=True
PYTHON="python3"
EXTRACTOR="JavaExtractor/extract.py"
NUM_THREADS=1
REMOVE="rm"
# REMOVE="del"

def preprocess_dir(path:str):

    for f in os.listdir(path + "/src"):
        # if f[-13: -5] != "_mutants": # analyize original
        #     if (not CONTINUE_PROCESS) or (not os.path.exists(os.path.join(path, "original.test.c2v"))):
        #         f = os.path.join(path, "src", f)
        #         os.system(
        #             PYTHON + " " + EXTRACTOR + " --file " + f + " --max_path_length 8 --max_path_width 2 --num_threads " + str(
        #                 NUM_THREADS) +
        #             " --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > " + path + "/augtemporigin")
        #         os.system(PYTHON + " preprocess_test_batch.py --test_data " +
        #                   path + "/augtemporigin --max_contexts 200 --dict_file data/java14m/java14m --output_name " + path + "/original")
        #         os.remove(path + "/augtemporigin")
        # else: # analyize mutants
        f_name = f[:-5]
        if (not CONTINUE_PROCESS) or (not os.path.exists(os.path.join(path, f_name + ".test.c2v"))):
            f_absolutepath = os.path.join(path, "src", f)
            os.system(
                PYTHON + " " + EXTRACTOR + " --file " + f_absolutepath + " --max_path_length 8 --max_path_width 2 --num_threads " + str(
                    NUM_THREADS) +
                " --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > " + path + "/" + f_name + "tempmut")
            os.system(PYTHON + " preprocess_test_batch.py --test_data " +
                      path + "/" + f_name + "tempmut --max_contexts 200 --dict_file data/java14m/java14m --output_name " + path + "/" + f_name)
            os.remove(path + "/" + f_name + "tempmut")

if __name__ == "__main__":

    mypath = sys.argv[1]
    alldirs = [os.path.join(mypath, f).replace("\\", "/") for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]

    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(preprocess_dir, alldirs)