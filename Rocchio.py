"""
.. module:: Rocchio

Rocchio
******

:Description: TFIDFViewer

    Returns the result of a query over an index

:Authors:
    Nil Vilas Basil

:Version:

:Date:  17/10/2019
"""

from __future__ import print_function, division
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse
import numpy as np
from functools import reduce
from collections import Counter

__author__ = 'nil.vilas'


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get document terms frequency and overall terms document frequency
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = {}
    for (t, w), (_, df) in zip(file_tv, file_df):
        # Code filled
        tfdi = w / max_freq
        idfi = np.log2(dcount / df)
        tfidfw[t] = tfdi * idfi

    return normalize(tfidfw)


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    # Code filled

    div = np.sqrt(reduce(lambda y, x: x**2 + y, tw.values(), 0))
    return dict(map(lambda x: (x[0], x[1]/div), tw.items()))


def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


def parse_arguments():
    """
    Parses the input arguments

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nrounds', default=2,
                        help='Number of applications of Rocchios rule')
    parser.add_argument('--topdocs', default=5,
                        help='Number of top documents considered relevant and used for applying Rocchio at each round')
    parser.add_argument('--newterms', default=5,
                        help='Number of new terms to be kept in the new query')
    parser.add_argument('--alpha', default=5,
                        help='Weight of the old query in Rocchios rule')
    parser.add_argument('--beta', default=3,
                        help='Weight of the new terms in Rocchios rule')
    parser.add_argument('--query', default=None,
                        nargs=argparse.REMAINDER, help='List of words to search')

    return parser.parse_args()


def execute_query(query, k, s):
    """
    Executes a the 'query' on client 's0 and returns the
    'k' most relevant results.

    Arguments:
        query {string} -- Query to be executed
        k {[type]} -- Number of relevintant results to return
        s {[type]} -- Search client

    Returns:
        responses of the k most relevant results (with the
        document id)
    """
    q = Q('query_string', query=query[0])
    for i in range(1, len(query)):
        q &= Q('query_string', query=query[i])
    print(q)
    s = s.query(q)
    response = s[0:k].execute()
    return response


def get_response_tfidf(client, index, response):
    tfidfs = [toTFIDF(client, index, r.meta.id) for r in response]
    res = {}
    for i in range(len(tfidfs)):
        for key in tfidfs[i]:
            res[key] = tfidfs[i][key]
            for j in range(i+1, len(tfidfs)):
                if tfidfs[j].get(key) is not None:
                    res[key] += tfidfs[j][key]
                    del tfidfs[j][key]
    return res


def get_terms(tfidf, R, kdocs, beta):
    s = [(k, float(beta*v/kdocs))
         for k, v in sorted(tfidf.items(), key=lambda x: x[1], reverse=True)]
    return dict(s[0:R])


def new_query(query, terms, a):
    q = {k: float(v*a) for k, v in query.items()}
    cdict = Counter(q) + Counter(terms)
    return dict(cdict)


def parse_query(q):
    query = {}
    for t in q:
        q_term = t.split('^')
        if(len(q_term) > 1):
            query[q_term[0]] = float(q_term[1])
        else:
            query[q_term[0]] = float(1)
    return query


def stringify_query(q):
    query = []
    for key in q:
        query.append(key + '^' + str(q[key]))
    print(query)
    return query


def print_response(response):
    for r in response:
        print(f'ID= {r.meta.id} SCORE={r.meta.score}')
        print(f'PATH= {r.path}')
        print(f'TEXT: {r.text[:50]}')
        print('-----------------------------------------------------------------')


if __name__ == '__main__':
    args = parse_arguments()

    index = args.index
    query = parse_query(args.query)
    nrounds = int(args.nrounds)
    k = int(args.topdocs)
    R = int(args.newterms)
    a = float(args.alpha)
    b = float(args.beta)

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)
        if query is not None:
            print(f'Using Rocchio to find the best results in {index}')
            response = []
            for i in range(nrounds):
                print(query)
                response = execute_query(stringify_query(query), k, s)
                docTFIDF = get_response_tfidf(client, index, response)
                terms = get_terms(docTFIDF, R, k, b)
                query = new_query(query, terms, a)
                print_response(response)
            print_response(response)
        else:
            print('No query parameters passed')
    except NotFoundError:
        print(f'Index {index} does not exist')
