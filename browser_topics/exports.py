"""Downloadable results.

`SPECS.md` section 7 fixes the columns of each file. Every function here returns a frame or bytes.
No function touches the file system, because Pyodide has no disk.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from browser_topics.metrics import topic_similarity

if TYPE_CHECKING:
    from browser_topics.config import AppConfig
    from browser_topics.result import TopicModelResult

__all__ = [
    "ZIP_README",
    "config_json",
    "documents_topics_frame",
    "project_zip",
    "to_csv_bytes",
    "topic_similarity_frame",
    "topic_terms_frame",
    "topics_frame",
]

TOP_TERM_COUNT = 10
"""Terms per topic in `topics.csv` and `topic_terms.csv`."""

ZIP_README = """Browser Topic Explorer export

documents_topics.csv   one row per modelled document, with its topic shares
topics.csv             one row per topic, with its name, prevalence and top terms
topic_terms.csv        long format, one row per topic and term
topic_similarity.csv   cosine similarity between every pair of topics
config.json            the settings that produced this result

Topic scores are shares. Each document's scores sum to 1.
The projection columns place a document on the 2-D map. The axes carry no meaning.
"""


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    r"""Encode a frame as UTF-8 CSV bytes for a download.

    >>> to_csv_bytes(pd.DataFrame({"a": [1]}))
    b'a\n1\n'
    """
    return frame.to_csv(index=False).encode("utf-8")


def documents_topics_frame(result: TopicModelResult, include_text: bool = False) -> pd.DataFrame:
    """Build `documents_topics.csv` with the columns of `SPECS.md` section 7.

    >>> from browser_topics.result import _example_result
    >>> frame = documents_topics_frame(_example_result())
    >>> frame.columns.tolist()[:3]
    ['document_id', 'dominant_topic_id', 'dominant_topic_name']
    >>> "text" in frame.columns
    False
    >>> documents_topics_frame(_example_result(), include_text=True).columns.tolist()[1]
    'text'
    >>> frame["topic_0_score"].round(2).tolist()
    [1.0, 0.0, 0.5]
    """
    columns: dict[str, Any] = {"document_id": result.document_ids}
    if include_text:
        columns["text"] = result.documents
    for name in result.metadata.columns:
        columns[str(name)] = result.metadata[name].to_numpy()
    columns["dominant_topic_id"] = result.dominant_topic
    columns["dominant_topic_name"] = [result.topic_names[index] for index in result.dominant_topic]
    columns["dominant_topic_score"] = result.dominant_topic_score
    for index in range(result.n_topics):
        columns[f"topic_{index}_score"] = result.document_topic[:, index]
    columns["projection_x"] = result.document_xy[:, 0]
    columns["projection_y"] = result.document_xy[:, 1]
    return pd.DataFrame(columns)


def topics_frame(result: TopicModelResult, top_n: int = TOP_TERM_COUNT) -> pd.DataFrame:
    """Build `topics.csv` with the columns of `SPECS.md` section 7.

    >>> from browser_topics.result import _example_result
    >>> frame = topics_frame(_example_result(), top_n=2)
    >>> frame.columns.tolist()
    ['topic_id', 'topic_name', 'topic_auto_label', 'prevalence', 'x', 'y', 'top_terms']
    >>> frame["top_terms"].tolist()
    ['alpha, beta', 'gamma, beta']
    """
    terms = result.top_terms(top_n)
    return pd.DataFrame(
        {
            "topic_id": range(result.n_topics),
            "topic_name": result.topic_names,
            "topic_auto_label": result.topic_auto_labels,
            "prevalence": result.topic_prevalence,
            "x": result.topic_xy[:, 0],
            "y": result.topic_xy[:, 1],
            "top_terms": [", ".join(row) for row in terms],
        }
    )


def topic_terms_frame(result: TopicModelResult, top_n: int = TOP_TERM_COUNT) -> pd.DataFrame:
    """Build the long-format `topic_terms.csv` of `SPECS.md` section 7.

    >>> from browser_topics.result import _example_result
    >>> frame = topic_terms_frame(_example_result(), top_n=2)
    >>> frame.columns.tolist()
    ['topic_id', 'topic_name', 'rank', 'term', 'weight_raw', 'weight_normalized']
    >>> frame["rank"].tolist()
    [1, 2, 1, 2]
    >>> frame["term"].tolist()
    ['alpha', 'beta', 'gamma', 'beta']
    """
    limit = min(top_n, len(result.feature_names))
    order = np.argsort(-result.topic_term, axis=1)[:, :limit]
    rows = []
    for topic_id, indices in enumerate(order.tolist()):
        for rank, term_index in enumerate(indices, start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_name": result.topic_names[topic_id],
                    "rank": rank,
                    "term": result.feature_names[term_index],
                    "weight_raw": float(result.topic_term_raw[topic_id, term_index]),
                    "weight_normalized": float(result.topic_term[topic_id, term_index]),
                }
            )
    return pd.DataFrame(rows)


def topic_similarity_frame(result: TopicModelResult) -> pd.DataFrame:
    """Build `topic_similarity.csv` with one row per distinct topic pair.

    >>> from browser_topics.result import _example_result
    >>> frame = topic_similarity_frame(_example_result())
    >>> frame.columns.tolist()
    ['topic_a_id', 'topic_a_name', 'topic_b_id', 'topic_b_name', 'cosine_similarity']
    >>> len(frame)
    1
    """
    similarity = topic_similarity(result.topic_term)
    rows_index, columns_index = np.triu_indices(result.n_topics, k=1)
    return pd.DataFrame(
        {
            "topic_a_id": rows_index,
            "topic_a_name": [result.topic_names[index] for index in rows_index],
            "topic_b_id": columns_index,
            "topic_b_name": [result.topic_names[index] for index in columns_index],
            "cosine_similarity": similarity[rows_index, columns_index],
        }
    )


def config_json(config: AppConfig, topic_names: list[str] | None = None) -> bytes:
    """Serialize the configuration for `config.json`.

    The language always leaves as `es`, never as the accepted alias `sp`.

    >>> from browser_topics.config import AppConfig
    >>> import json
    >>> payload = json.loads(config_json(AppConfig(language="sp")))
    >>> payload["language"]
    'es'
    >>> payload["model"]["model_type"]
    'nmf'
    >>> sorted(payload)
    ['app_version', 'language', 'model', 'preprocess', 'stop_words', 'topic_names']
    """
    payload = config.model_dump()
    payload["topic_names"] = topic_names or []
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def project_zip(result: TopicModelResult, config: AppConfig, include_text: bool = False) -> bytes:
    """Bundle every export into `project.zip`.

    `SPECS.md` section 7 names the six entries. The bundle is built in memory with the standard
    `zipfile` module.

    >>> import zipfile, io
    >>> from browser_topics.config import AppConfig
    >>> from browser_topics.result import _example_result
    >>> data = project_zip(_example_result(), AppConfig())
    >>> zipfile.ZipFile(io.BytesIO(data)).namelist()
    ['documents_topics.csv', 'topics.csv', 'topic_terms.csv',
     'topic_similarity.csv', 'config.json', 'README.txt']
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "documents_topics.csv", to_csv_bytes(documents_topics_frame(result, include_text))
        )
        archive.writestr("topics.csv", to_csv_bytes(topics_frame(result)))
        archive.writestr("topic_terms.csv", to_csv_bytes(topic_terms_frame(result)))
        archive.writestr("topic_similarity.csv", to_csv_bytes(topic_similarity_frame(result)))
        archive.writestr("config.json", config_json(config, result.topic_names))
        archive.writestr("README.txt", ZIP_README)
    return buffer.getvalue()
