import numpy as np


def test_inmemory_vector_collection_filters_updates_and_deletes():
    from engine.speaker.vector_store import InMemoryVectorCollection

    collection = InMemoryVectorCollection("test")
    collection.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"session_id": "s1", "count": 1},
            {"session_id": "s2", "count": 1},
        ],
    )

    result = collection.query(
        query_embeddings=[[0.9, 0.1]],
        n_results=3,
        where={"session_id": "s1"},
    )

    assert result["ids"] == [["a"]]
    assert result["metadatas"][0][0]["session_id"] == "s1"

    collection.update(
        ids=["a"],
        embeddings=[np.array([0.0, 1.0], dtype=np.float32)],
        metadatas=[{"session_id": "s1", "count": 2}],
    )
    updated = collection.get(ids=["a"], include=["embeddings", "metadatas"])
    assert updated["metadatas"][0]["count"] == 2
    assert updated["embeddings"][0][1] > 0.99

    collection.delete(where={"session_id": "s1"})
    assert collection.query(
        query_embeddings=[[0.0, 1.0]],
        n_results=3,
        where={"session_id": "s1"},
    )["ids"] == [[]]
    assert collection.count() == 1
