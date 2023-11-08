#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriever
"""

import faiss
import logging
import numpy as np
import os
import pickle
import torch

from typing import List, Tuple

from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.client import IndexClient

logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)

class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.init_index(768)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # scores, indexes, embds = self.index.search_and_reconstruct(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        # result = [(db_ids[i], scores[i], embds[i]) for i in range(len(db_ids))]
        result = [(db_ids[i], scores[i], None) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"

# class DenseFlatIndexer(DenseIndexer):
#     def __init__(self, buffer_size: int = 50000):
#         super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
#         self.use_IVF = True

#     def init_index(self, vector_sz: int):
#         if not self.use_IVF:
#             self.index = faiss.IndexFlatIP(vector_sz)
#         else:
#             flat_index = faiss.IndexFlatIP(vector_sz)
#             self.index = faiss.IndexIVFFlat(flat_index, vector_sz, 1024)
#             self.index.nprobe = 8

#     def index_data(self, data: List[Tuple[object, np.array]]):
#         n = len(data)
#         # n = self.buffer_size # Fast work around

#         # indexing in batches is beneficial for many faiss index types
#         for i in range(0, n, min(self.buffer_size,n) ):
#             db_ids = [t[0] for t in data[i : i + self.buffer_size]]
#             vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
#             vectors = np.concatenate(vectors, axis=0)
#             total_data = self._update_id_mapping(db_ids)
#             if self.use_IVF and not self.index.is_trained:
#                 self.index.train(vectors)
#             self.index.add(vectors)
#             logger.info("data indexed %d", total_data)

#         indexed_cnt = len(self.index_id_to_db_id)
#         logger.info("Total data indexed %d", indexed_cnt)

#     def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
#         scores, indexes, vectors = self.index.search_and_reconstruct(query_vectors, top_docs)
#         # scores, indexes = self.index.search(query_vectors, top_docs)
#         # convert to external ids
#         db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
#         result = [(db_ids[i], scores[i], vectors[i]) for i in range(len(db_ids))]
#         return result

#     def get_index_name(self):
#         return "flat_index"

# Load that mapping !!
class DenseDisIVFIndex(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseDisIVFIndex, self).__init__(buffer_size=buffer_size)
        self.client = None
        self.index_id = 'wiki'

    def init_client(self, args, local_rank = None):
        cfg = IndexCfg(
        index_builder_type=args.index_builder_type,
        dim=args.dim,
        train_num=args.train_num,
        centroids=args.centroids,
        metric=args.metric,
        nprobe=args.centroids,
        # nprobe=args.nprobe,
        index_storage_dir=args.index_storage_dir)
        self.client = IndexClient(args.discovery_config, cfg=cfg)
        # if local_rank is not None:
        #     self.client = IndexClient(args.discovery_config + '_' + str(torch.cuda.current_device() // 2), cfg=cfg)
        # else:
        #     self.client = IndexClient(args.discovery_config, cfg=cfg)

        with open(args.db_to_index, 'rb') as f:
            self.index_id_to_db_id =  pickle.load(f)
        

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes, vectors = self.client.search(query_vectors, top_docs, self.index_id, return_embeddings=True)
        # scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[int(i)] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i], vectors[i]) for i in range(len(db_ids))]
        return result
    


        

class DenseHNSWFlatIndexer(DenseIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e9,
        store_n: int = 512,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.store_n = store_n
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.phi = 0

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once," "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item[0:2]
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = phi

        # indexing in batches is beneficial for many faiss index types
        bs = int(self.buffer_size)
        for i in range(0, n, bs):
            db_ids = [t[0] for t in data[i : i + bs]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + bs]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
            self.train(hnsw_vectors)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info("data indexed %d", len(self.index_id_to_db_id))
        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def train(self, vectors: np.array):
        pass

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize(file)
        # to trigger exception on subsequent indexing
        self.phi = 1

    def get_index_name(self):
        return "hnsw_index"


class DenseHNSWSQIndexer(DenseHNSWFlatIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e10,
        store_n: int = 128,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        super(DenseHNSWSQIndexer, self).__init__(
            buffer_size=buffer_size,
            store_n=store_n,
            ef_search=ef_search,
            ef_construction=ef_construction,
        )

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWSQ(vector_sz + 1, faiss.ScalarQuantizer.QT_8bit, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def train(self, vectors: np.array):
        self.index.train(vectors)

    def get_index_name(self):
        return "hnswsq_index"
