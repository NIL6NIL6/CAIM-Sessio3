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

__author__ = 'nil.vilas'

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id

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

    tfidfw = []
    for (t, w),(_, df) in zip(file_tv, file_df):
        #Code filled
        tfdi = w / max_freq
        idfi = np.log2(dcount / df)
        tfidfw.append((t, tfdi * idfi))

    return normalize(tfidfw)

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    #Code filled
    div = np.sqrt(reduce(lambda y, x: x[1]**2 + y, tw, 0))
    return list(map(lambda x: (x[0], x[1]/div), tw))

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
	parser.add_argument('--nrounds', default=2, help='Number of applications of Rocchios rule')
	parser.add_argument('--topdocs', default=5, help='Number of top documents considered relevant and used for applying Rocchio at each round')
	parser.add_argument('--newterms', default=5, help='Number of new terms to be kept in the new query')
	parser.add_argument('--alpha', default=5, help='Weight of the old query in Rocchios rule')
	parser.add_argument('--beta', default=3, help='Weight of the new terms in Rocchios rule')
	parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')
	
	return parser.parse_args()
	

if __name__ == '__main__':	
	args = parse_arguments()
	
	index = args.index
	query = args.query
	nrounds = args.nrounds
	k = args.topdocs
	R = args.newterms
	a = args.alpha
	b = args.beta
	
	try:
		client = Elasticsearch()
		s = Search(using=client, index=index)

		if query is not None:
			print(f'Using Rocchio to find the best results in {index}')
		else:
			print('No query parameters passed')
	except NotFoundError:
		print(f'Index {index} does not exist')
    
    
