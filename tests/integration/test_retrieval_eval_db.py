"""Integration test: the retrieval eval's ingest + retriever wiring against
a real test database, with the offline FakeEmbedder."""

from app.schemas.search import MAX_TOP_K
from app.services.embedding import FakeEmbedder
from eval.run import DEFAULT_DATASET as CORPUS_PATH
from eval.run import load_dataset
from eval.run_retrieval import ingest_corpus, make_retriever


def test_ingested_corpus_is_retrievable_by_corpus_id(db_session):
    corpus = load_dataset(CORPUS_PATH)[:5]
    embedder = FakeEmbedder()

    id_map = ingest_corpus(db_session, corpus, embedder)
    retrieved = make_retriever(db_session, embedder, id_map)("any question")

    assert sorted(id_map.values()) == sorted(item.id for item in corpus)
    assert len(retrieved) == min(len(corpus), MAX_TOP_K)
    assert {email_id for email_id, _ in retrieved} == set(id_map.values())
    scores = [score for _, score in retrieved]
    assert scores == sorted(scores, reverse=True)
