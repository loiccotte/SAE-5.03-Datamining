"""
BM25Vectorizer — Alternative à TfidfVectorizer basée sur le scoring BM25.
Interface compatible sklearn : méthodes fit, transform, fit_transform, get_feature_names_out.
La clé 'tfidf_vectorizer' des fichiers pkl contiendra cet objet quand vect_lib='bm25'.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Vectorizer:
    """
    Vectoriseur BM25 (Okapi BM25).
    Paramètres BM25 : k1=1.5, b=0.75 (valeurs standard Robertson et al.)
    Accepte les mêmes kwargs que CountVectorizer (ngram_range, min_df, max_df, token_pattern, lowercase).
    """

    def __init__(self, k1=1.5, b=0.75, **cv_kwargs):
        self.k1 = k1
        self.b = b
        self._cv = CountVectorizer(**cv_kwargs)
        self._idf = None
        self._avgdl = None

    def fit(self, raw_documents):
        counts = self._cv.fit_transform(raw_documents)
        self._compute_idf_avgdl(counts)
        return self

    def transform(self, raw_documents):
        counts = self._cv.transform(raw_documents)
        return self._bm25_weights(counts)

    def fit_transform(self, raw_documents):
        counts = self._cv.fit_transform(raw_documents)
        self._compute_idf_avgdl(counts)
        return self._bm25_weights(counts)

    def _compute_idf_avgdl(self, counts):
        n_docs = counts.shape[0]
        dl = np.asarray(counts.sum(axis=1)).flatten()
        self._avgdl = dl.mean() if dl.mean() > 0 else 1.0
        df = np.asarray((counts > 0).sum(axis=0)).flatten()
        self._idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def _bm25_weights(self, counts):
        counts = counts.astype(float).tocsr()
        dl = np.asarray(counts.sum(axis=1)).flatten()
        rows, cols = counts.nonzero()
        tf = np.asarray(counts[rows, cols]).flatten()
        dl_norm = dl[rows] / self._avgdl
        denom = tf + self.k1 * (1 - self.b + self.b * dl_norm)
        bm25 = (tf * (self.k1 + 1)) / denom * self._idf[cols]
        return sp.csr_matrix((bm25, (rows, cols)), shape=counts.shape)

    def get_feature_names_out(self):
        return self._cv.get_feature_names_out()
