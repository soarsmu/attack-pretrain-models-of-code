import json
import requests
import common_adversarial
import numpy as np

ONLY_VARIABLE = "VUNK"

def guard_by_n2p(code_sample_with_vars, is_word_in_vocab_func):
    variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
    if variables == "":
        return code
    else:
        variables = variables.lower().split(",")

    existed_variables = [v for v in variables if is_word_in_vocab_func(v)]
    # get all tokens
    contexts = code.split(" ")
    contexts = [c.split(",") for c in contexts[1:] if c != ""]
    name_src, _, name_dst = zip(*contexts)
    id_to_name = list(set(name_src + name_dst))
    names_to_id = {name: i for i, name in enumerate(id_to_name)}

    assign_field = [{"v": id, "inf" if name in existed_variables else "giv": name}
                    for name, id in names_to_id.items()]
    query_field = [{"a": names_to_id[p[0]], "b": names_to_id[p[2]], "f2": p[1]}
                   for p in contexts]
    query_field.append({"cn":"!=", "n":list(range(len(id_to_name)))})

    # build request
    request = {"params": {"assign": assign_field, "query": query_field},
               "method": "infer",
               "id": "1",
               "jsonrpc": "2.0"}

    response = requests.post('http://pomela4.cslcs.technion.ac.il:5745', json.dumps(request))

    # parse response
    response = json.loads(response.text)
    replace_variables = {id_to_name[r["v"]]: r["inf"].lower() for r in response["result"]
                         if "inf" in r and id_to_name[r["v"]] != r["inf"].lower()}
    # method_name = code.split(" ")[0]
    # print("method:", method_name)
    # print("total vars:", len(variables), variables)
    for original_var, new_var in replace_variables.items():
        # print("replace:", original_var, "(", is_word_in_vocab_func(original_var), ")",
        #       "with", new_var, "(", is_word_in_vocab_func(new_var), ")")
        code = code.replace(" " + original_var + ",", " " + new_var + ",") \
            .replace("," + original_var + " ", "," + new_var + " ")

    return code

def guard_by_vunk(code_sample_with_vars):
    variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
    if variables != "":
        variables = variables.lower().split(",")
        for original_var in variables:
            code = code.replace(" " + original_var + ",", " " + ONLY_VARIABLE + ",") \
                .replace("," + original_var + " ", "," + ONLY_VARIABLE + " ")

    return code

def guard_by_distance(code_sample_with_vars, is_word_in_vocab_func, get_embed_func, threshold):
    variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
    variables = common_adversarial.get_all_vars(variables)
    tokens = common_adversarial.get_all_tokens(code)

    if not variables:
        return code

    existed_variables = [v for v in variables if is_word_in_vocab_func(v)]

    if not existed_variables:
        return code

    # if len(existed_variables) == 1:
    #     return common_adversarial.replace_var_in_code(code, existed_variables[0], ONLY_VARIABLE)

    both = list(set(existed_variables + list(tokens)))

    exist_tokens = [v for v in both if is_word_in_vocab_func(v)]
    # embed = get_embed_func(exist_tokens)
    # embed = {v: e for v, e in zip(exist_tokens, embed)}
    embed = {v:get_embed_func(v) for v in exist_tokens}



    embed_sum = np.sum(list(embed.values()), axis=0)
    # embed = {v: e - (embed_sum / len(embed)) for v, e in embed.items()}

    dist = {v : np.linalg.norm(embed[v] - ((embed_sum - embed[v]) / (len(embed) - 1)),ord=2) for v in existed_variables}
    bad_var = max(dist, key=lambda v: dist[v])

    # determine if bad_var is really bad
    # avg_embed = np.mean([embed[v] for v in exist_tokens if v != bad_var], axis=0)
    # distance_distribution = [np.linalg.norm(embed[v] - avg_embed,ord=2) for v in exist_tokens if v != bad_var]
    # distance_mean = np.mean(distance_distribution)
    # distance_std = np.std(distance_distribution)

    # is_truly_bad = abs(np.linalg.norm(embed[bad_var] - avg_embed,ord=2) - distance_mean) > 3 * distance_std or \
    #                abs(np.linalg.norm(embed[bad_var] - avg_embed, ord=2) - distance_mean) < 2 * distance_std

    is_truly_bad = dist[bad_var] > threshold

    # print("method:", code.split(" ")[0], "picked:", bad_var, "reallybad:", is_truly_bad, dist, len(dist))
    # print("reallybad:", is_truly_bad, "(mean:{}, std:{},badist:{}".format(distance_mean,distance_std,np.linalg.norm(embed[bad_var] - avg_embed,ord=2)))

    return  common_adversarial.replace_var_in_code(code, bad_var, ONLY_VARIABLE) if is_truly_bad else code

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# def guard_by_pca(code_sample_with_vars, is_word_in_vocab_func, get_embed_func):
#     variables, code = common_adversarial.separate_vars_code(code_sample_with_vars)
#     variables = common_adversarial.get_all_vars(variables)
#     tokens = common_adversarial.get_all_tokens(code)
#
#     both = list(set(variables+list(tokens)))
#
#     # both = variables
#
#     exist_tokens = [v for v in both if is_word_in_vocab_func(v)]
#     if len(exist_tokens)==1:
#         return code
#     exist_embed = [get_embed_func(v) for v in exist_tokens]
#
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(exist_embed)
#
#     # x, y = zip(*principalComponents)
#     # fig, ax = plt.subplots()
#     for (x,y), name in zip(principalComponents, exist_tokens):
#         al = 1 if name in variables else 0.5
#         name = name + ("_V" if name in variables else "")
#         plt.scatter(x, y, label=name, alpha=al, edgecolors='none')
#
#     plt.title(code.split(" ")[0])
#     plt.legend()
#
#     plt.savefig("pca/" + datetime.now().strftime('%Y%m%d_%H_%M_%S')+".png")
#     plt.clf()
#
#     # fig = plt.figure(figsize=(8, 8))
#     # ax = fig.add_subplot(1, 1, 1)
#     # ax.set_xlabel('Principal Component 1', fontsize=15)
#     # ax.set_ylabel('Principal Component 2', fontsize=15)
#     # ax.set_title('2 component PCA', fontsize=20)
#     # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#     # colors = ['r', 'g', 'b']
#     # for target, color in zip(targets, colors):
#     #     indicesToKeep = finalDf['target'] == target
#     #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#     #                , finalDf.loc[indicesToKeep, 'principal component 2']
#     #                , c=color
#     #                , s=50)
#     # ax.legend(targets)
#     # ax.grid()
#
#     return code