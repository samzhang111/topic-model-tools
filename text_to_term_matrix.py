import scipy as sp
import numpy as np
import textacy
from collections import defaultdict

def text_to_term_matrix(text, term2id, verbose=True):
    """This method constructs a sparse matrix from a text, by calculating
    tf-idf based off existing document frequencies.
    """
    terms_list = textacy.doc.Doc(text, lang='en').to_terms_list(ngrams=1, named_entities=True, as_strings=True)
    term_counts = defaultdict(int)
    not_found = []
    N = 0
    for term in terms_list:
        try:
            term_id = term2id[term]
            term_counts[term_id] += 1
            N += 1
        except KeyError:
            not_found.append(term)

    tfidfs = {}
    l2_norm_denominator = 0.0
    for term_id, count in term_counts.items():
        tf = count / float(N)

        # smoothing as done in https://github.com/chartbeat-labs/textacy/blob/master/textacy/vsm.py#L144
        idf = np.log(1 / dfs[term_id]) + 1

        tfidfs[term_id] = tf * idf
        l2_norm_denominator += (tf * idf) ** 2

    l2_norm_denominator = np.sqrt(l2_norm_denominator)
    if verbose:
        print('Normalizing by L2 norm', l2_norm_denominator)
    for term_id, tfidf in tfidfs.items():
        tfidfs[term_id] = tfidf / l2_norm_denominator

    if verbose and not_found:
        print('Not found: ', ', '.join(not_found))

    # Building sparse matrix
    data = []
    rows = []
    columns = []
    for term_id, tfidf in tfidfs.items():
        data.append(tfidf)
        columns.append(term_id)
        rows.append(0)

    term_matrix = sp.coo_matrix((data, (rows, columns)), shape=[1, 200000])
    return term_matrix

