`ir_datasets`: Python API
=========================

Dataset objects
---------------

Datasets can be obtained through `ir_datasets.load("dataset-id")` or constructed with `ir_datasets.create_dataset(...)`. Dataset objects provide the following methods:

#### `dataset.has_docs() -> bool`

Returns `True` if this dataset supports `dataset.docs_*` methods.

#### `dataset.has_queries() -> bool`

Returns `True` if this dataset supports `dataset.queries_*` methods.

#### `dataset.has_qrels() -> bool`

Returns `True` if this dataset supports `dataset.qrels_*` methods.

#### `dataset.has_scoreddocs() -> bool`

Returns `True` if this dataset supports `dataset.scoreddocs_*` methods.

#### `dataset.has_docpairs() -> bool`

Returns `True` if this dataset supports `dataset.docpairs_*` methods.

#### `dataset.docs_count() -> int`

Returns the number of documents in the collection.

#### `dataset.docs_iter() -> iter[namedtuple]`

Returns an iterator of `namedtuple`s, where each item is a document in the collection.

This iterator supports fancy slicing (with some limitations):

`

# First 10 documents

dataset.docs_iter()[:10]

# Last 10 documents

dataset.docs_iter()[-10:]

# Every 2 documents

dataset.docs_iter()[::2]

# Every 2 documents, starting with the first document

dataset.docs_iter()[1::2]

# The first half of the collection

dataset.docs_iter()[:1/2]

# The middle third of collection

dataset.docs_iter()[1/3:2/3]

`

Note that the fancy slicing mechanics are faster and more sophisticated than `itertools.islice`; documents are not processed if they are skipped.

#### `dataset.docs_cls() -> type`

Returns the `NamedTuple` type that the `docs_iter` returns. The available fields and type information can be found with `_fields` and `__annotations__`:

`dataset.docs_cls()._fields``('doc_id', 'title', 'doi', 'date', 'abstract')``dataset.docs_cls().__annotations__``

{{

  'doc_id': str,

  'title': str,

  'doi': str,

  'date': str,

  'abstract': str

}}

`

#### `dataset.docs_store() -> docstore`

Returns a [docstore object](https://ir-datasets.com/python.html#docstore) for this dataset, which enables fast lookups by `doc_id`.

#### `dataset.docs_lang() -> str`

Returns the two-character [ISO 639-1 language code](https://en.wikipedia.org/wiki/ISO_639-1) (e.g., "en" for English) of the documents in this collection. Returns None if there are multiple languages, a language not represented by an ISO 639-1 code, or the language is otherwise unknown.

#### `dataset.docs_metadata() -> dict`

Returns available metadata about the docs from this dataset (e.g., count).

#### `dataset.queries_count() -> int`

Returns the number of queries in the collection.

#### `dataset.queries_iter() -> iter[namedtuple]`

Returns an iterator over namedtuples representing queries in the dataset.

#### `dataset.queries_cls() -> type`

Returns the type of the namedtuple returned by `queries_iter`, including `_fields` and `__annotations__`.

#### `dataset.queries_lang() -> str`

Returns the two-character [ISO 639-1 language code](https://en.wikipedia.org/wiki/ISO_639-1) (e.g., "en" for English) of the queries. Returns None if there are multiple languages, a language not represented by an ISO 639-1 code, or the language is otherwise unknown. Note that some datasets include translations as different query fields.

#### `dataset.queries_metadata() -> dict`

Returns available metadata about the queries from this dataset (e.g., count).

#### `dataset.qrels_count() -> int`

Returns the number of qrels in the collection.

#### `dataset.qrels_iter() -> iter[namedtuple]`

Returns an iterator over namedtuples representing query relevance assessments in the dataset.

#### `dataset.qrels_cls() -> type`

Returns the type of the namedtuple returned by `qrels_iter`, including `_fields` and `__annotations__`.

#### `dataset.qrels_defs() -> dict[int, str]`

Returns a mapping between relevance levels and a textual description of what the level represents. (E.g., 0 represting not relevant, 1 representing possibly relevant, 2 representing definitely relevant.)

#### `dataset.qrels_dict() -> dict[str, dict[str, int]]`

Returns a dict of dicts representing all qrels for this collection. Note that this will load all qrels into memory. The outer dict key is the `query_id` and the inner key is the `doc_id`. This is useful in tools such as [pytrec_eval](https://github.com/cvangysel/pytrec_eval).

#### `dataset.qrels_metadata() -> dict`

Returns available metadata about the qrels from this dataset (e.g., count).

#### `dataset.scoreddocs_count() -> int`

Returns the number of scoreddocs in the collection.

#### `dataset.scoreddocs_iter() -> iter[namedtuple]`

Returns an iterator over namedtuples representing scored docs (e.g., initial rankings for re-ranking tasks) in the dataset.

#### `dataset.scoreddocs_cls() -> type`

Returns the type of the namedtuple returned by `scoreddocs_iter`, including `_fields` and `__annotations__`.

#### `dataset.scoreddocs_metadata() -> dict`

Returns available metadata about the scoreddocs from this dataset (e.g., count).

#### `dataset.docpairs_count() -> int`

Returns the number of docpairs in the collection.

#### `dataset.docpairs_iter() -> iter[namedtuple]`

Returns an iterator over namedtuples representing doc pairs (e.g., training pairs) in the dataset.

#### `dataset.docpairs_cls() -> type`

Returns the type of the namedtuple returned by `docpairs_iter`, including `_fields` and `__annotations__`.

#### `dataset.docpairs_metadata() -> dict`

Returns available metadata about the docpairs from this dataset (e.g., count).

#### `dataset.qlogs_count() -> int`

Returns the number of query log records in the collection.

#### `dataset.qlogs_iter() -> iter[namedtuple]`

Returns an iterator over namedtuples representing query log records in the dataset.

#### `dataset.qlogs_cls() -> type`

Returns the type of the namedtuple returned by `qlogs_iter`, including `_fields` and `__annotations__`.

#### `dataset.qlogs_metadata() -> dict`

Returns available metadata about the qlogs from this dataset (e.g., count).

Docstore objects
----------------

Docstores enable fast lookups of documents by their `doc_id`.

The implementation depends on the dataset. For small datasets, a simple index structure is built on disk to enable fast lookups. For large datasets, you wouldn't want to make a copy of the collection, so lookups are accelerated by taking advantage of the source file structure and decompression checkpoints.

For small datasets, docstores also enable faster iteration and fancy slicing. In some cases, a docstore instance is automatically generated during the first call to `docs_iter` to enable faster iteration in the future.

#### `docstore.get(doc_id: str) -> namedtuple`

Gets a single document by `doc_id`. Returns a single namedtuple or throws a KeyError if the document it not in the collection.

#### `docstore.get_many(doc_ids: iter[str]) -> dict[str, namedtuple]`

Gets documents whose IDs appear in `doc_ids`. Returns a dict mapping string IDs to namedtuple. Missing documents will not appear in the dictionary.

#### `docstore.get_many_iter(doc_ids: iter[str]) -> iter[namedtuple]`

Returns an iterator over documents whose IDs appear in `doc_ids`. The order of the documents is not guaranteed to be the same as doc_ids. (This is to allow implementations to optmize the order in which documents are retrieved from disk.) Missing documents will not appear in the iterator.