import pandas as pd
from math import log
# TODO page 312, 178 for skip pointers
# TODO find same set between all three, if over 10, calculate that


def prob_word_doc(selection, mu_val_set):
    doc_info_csv = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)
    # between 0-1
    # lambda_val = 1
    mu_val = mu_val_set
    # number of documents
    N = int(doc_info_csv.loc[0][0])
    # average document length
    avdl = int(doc_info_csv.loc[0][1])
    # total word occurrences in collection
    collection_c = N * avdl

    doc_score_dict = {}


    for term, row in selection.iterrows():
        doc_appear_list = eval(row[2])
        # how many times a word appears in collection (sum of appears column)
        c_qi = int(row[1])

        for doc in doc_appear_list:
            split_val = doc.split(":")
            doc_name = split_val[0]
            # frequency of term in document
            f_qi_d = int(split_val[1])
            # length of document
            dl = doc_info_csv[doc_info_csv[1] == doc_name][2].values[0]
            # frequency of term in document + mu value * term occurrences
            # in collection over total collection size
            numerator = f_qi_d + (mu_val*(c_qi/collection_c))
            denominator = dl + mu_val
            if doc_name in doc_score_dict.keys():
                doc_score_dict[doc_name] += log(numerator/denominator)
            else:
                doc_score_dict[doc_name] = log(numerator/denominator)

    results = pd.DataFrame.from_dict(doc_score_dict, orient='index')
    x_row = results.sort_values(by=[0], ascending=True)[:10]
    return x_row
