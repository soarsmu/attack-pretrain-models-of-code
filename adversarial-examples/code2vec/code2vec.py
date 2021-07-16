import pickle

from batch_adversarial_search import BatchPredictorAdversarialBFS
from common import Config, VocabType
from argparse import ArgumentParser
from interactive_predict import InteractivePredictor
from interactive_predict_adversarial_search import InteractivePredictorAdvMonoSearch, \
    InteractivePredictorAdvSimilarSearch, InteractivePredictorAdversarialBFS
from model import Model
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)
    parser.add_argument("-tfold", "--test_folder", dest="test_folder", action='store_true',
                        help="set this flag to do test on folder", required=False)

    parser.add_argument("-tadv", "--test_adversarial", dest="test_adversarial", action='store_true',
                        help="set this flag to do test with adversarial", required=False)
    parser.add_argument("-tadvdep", "--adversarial_depth", dest="adversarial_depth", default=2,
                        help="set this flag to determine the depth of BFS search", required=False)
    parser.add_argument("-tadvtopk", "--adversarial_topk", dest="adversarial_topk", default=2,
                        help="set this flag to determine the width of BFS search", required=False)
    parser.add_argument("-tadvtype", "--adversarial_type", dest="adversarial_type",
                        help="choose the type of attack. can be 'targeted' or 'non-targeted'", required=False)
    parser.add_argument("-tadvtrgt", "--adversarial_target", dest="adversarial_target",
                        help="choose desired target. or choose 'random-uniform'/'random-unigram' to select random target uniformly/unigramly", required=False)
    parser.add_argument("-tadvdead", "--adversarial_deadcode", dest="adversarial_deadcode", action='store_true', default=False,
                        help="set this flag to use dead-code attack (dataset preprocessed with deadcode required)", required=False)

    parser.add_argument("-grd", "--guard_input", dest="guard_input", type=float,
                        help="set this flag to use input guard",
                        required=False)

    parser.add_argument("-ldict", "--load_dict", dest="load_dict_path",
                        help="path to dict file", metavar="FILE", required=False)

    is_training = '--train' in sys.argv or '-tr' in sys.argv
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-w2v", "--save_word2v", dest="save_w2v",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-t2v", "--save_target2v", dest="save_t2v",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument('--save_w2v', dest='save_w2v', required=False,
                        help="save word (token) vectors in word2vec format")
    parser.add_argument('--save_t2v', dest='save_t2v', required=False,
                        help="save target vectors in word2vec format")
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a lower model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    config = Config.get_default_config(args)

    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        if not args.test_adversarial:
            model.train()
        else:
            model.adversarial_training()
    if args.save_w2v is not None:
        model.save_word2vec_format(args.save_w2v, source=VocabType.Token)
        print('Origin word vectors saved in word2vec text format in: %s' % args.save_w2v)
    if args.save_t2v is not None:
        model.save_word2vec_format(args.save_t2v, source=VocabType.Target)
        print('Target word vectors saved in word2vec text format in: %s' % args.save_t2v)
    if config.TEST_PATH and not args.data_path:
        if not args.test_folder and not args.test_adversarial:
            eval_results = model.evaluate(guard_input=args.guard_input)
            if eval_results is not None:
                results, precision, recall, f1 = eval_results
                print(results)
                print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        elif args.test_folder:
            eval_results = model.evaluate_folder()
            with open("total_results_" + config.TEST_PATH.replace("/","").replace("\\","") + ".pickle", 'wb') as handle:
                pickle.dump(eval_results, handle)
            # print(eval_results)
        elif args.test_adversarial:
            eval_results = model.evaluate_and_adverse(int(args.adversarial_depth),
                                                      int(args.adversarial_topk),
                                                      targeted_attack=args.adversarial_type=="targeted",
                                                      adversarial_target_word=args.adversarial_target,
                                                      deadcode_attack=args.adversarial_deadcode,
                                                      guard_input=args.guard_input,
                                                      data_dict_path=args.load_dict_path)
            with open("total_adversarial_results_" + config.TEST_PATH.replace("/", "").replace("\\", "") + ".pickle",
                      'wb') as handle:
                pickle.dump(eval_results, handle)

    if args.predict:
        if not args.test_adversarial:
            # manual adversarial search
            predictor = InteractivePredictor(config, model)
        else:
            predictor = InteractivePredictorAdversarialBFS(config, model,
            # predictor = BatchPredictorAdversarialBFS(config, model,
                                                           int(args.adversarial_topk),
                                                           int(args.adversarial_depth),
                                                           args.guard_input, False)
        # automatic search for something
        # predictor = InteractivePredictorAdvMonoSearch(config, model)
        # automatic search similar name
        # predictor = InteractivePredictorAdvSimilarSearch(config, model)
        predictor.predict()
    model.close_session()
