def test_store_can_import_without_eagerly_loading_retrieval_backends():
    import app.pipeline.store  # noqa: F401
