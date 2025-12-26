from __future__ import annotations

import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
)

import numpy as np
from vectorstores.docstore.base import AddableMixin, Docstore
from vectorstores.docstore.document import Document
from vectorstores.docstore.in_memory import InMemoryDocstore
from vectorstores.embedding import Embeddings
from vectorstores.base import VectorStore
from vectorstores.utils import DistanceStrategy, maximal_marginal_relevance


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))
    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return


class FAISS(VectorStore):
    """`Meta Faiss` vector store.

    To use, you must have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings.openai import OpenAIEmbeddings
            from langchain.vectorstores import FAISS

            embeddings = OpenAIEmbeddings()
            texts = ["FAISS is an important library", "LangChain supports FAISS"]
            faiss = FAISS.from_texts(texts, embeddings)

    """

    def __init__(
            self,
            embedding_function: Callable,
            embedding_query: Callable,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            relevance_score_fn: Optional[Callable[[float], float]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.MAX_INNER_PRODUCT,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.embedding_query = embedding_query
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if (
                self.distance_strategy != DistanceStrategy.MAX_INNER_PRODUCT
                and self._normalize_L2
        ):
            warnings.warn(
                "Normalizing L2 is not applicable for metric type: {strategy}".format(
                    strategy=self.distance_strategy
                )
            )

    def __add(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        faiss = dependable_faiss_import()

        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )

        _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]

        _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        _len_check_if_sized(documents, ids, "documents", "ids")

        # Add to the index.
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)

        # Add information to docstore and index.
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function(texts)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)

    def add_embeddings(
            self,
            text_embeddings: Iterable[Tuple[str, List[float]]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # Embed and create the documents.
        texts, embeddings = zip(*text_embeddings)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)

        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(doc.metadata.get(key) in value for key, value in filter.items()):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                   in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self.embedding_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
            self,
            embedding: List[float],
            *,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )

        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if all(
                        doc.metadata.get(key) in value
                        if isinstance(value, list)
                        else doc.metadata.get(key) == value
                        for key, value in filter.items()
                ):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.

        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        selected_scores = [scores[0][i] for i in mmr_selected]
        docs_and_scores = []
        for i, score in zip(selected_indices, selected_scores):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, score))
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")
        missing_ids = set(ids).difference(self.index_to_docstore_id.values())
        if missing_ids:
            raise ValueError(
                f"Some specified ids do not exist in the current store. Ids not found: "
                f"{missing_ids}"
            )

        reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
        index_to_delete = [reversed_index[id_] for id_ in ids]

        self.index.remove_ids(np.array(index_to_delete, dtype=np.int64))
        self.docstore.delete(ids)

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_docstore_id.items())
            if i not in index_to_delete
        ]
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True

    def merge_from(self, target: "FAISS") -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        for new_i, target_id, doc in full_info:
            self.index_to_docstore_id[new_i] = target_id
            if isinstance(self.docstore, InMemoryDocstore):
                self.docstore.add({target_id: doc})

