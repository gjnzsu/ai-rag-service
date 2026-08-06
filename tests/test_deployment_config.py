from pathlib import Path


def test_gke_persists_both_vector_and_lexical_indexes_under_data_volume():
    deployment = Path("k8s/deployment.yaml").read_text(encoding="utf-8")

    assert 'value: "/data/chroma_db"' in deployment
    assert "name: LEXICAL_DB_PATH" in deployment
    assert 'value: "/data/lexical.db"' in deployment
    assert "mountPath: /data" in deployment
