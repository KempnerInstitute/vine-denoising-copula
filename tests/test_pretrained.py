from vdc.pretrained import DEFAULT_PRETRAINED_MODEL_ID, list_pretrained_models, load_pretrained_manifest


def test_default_pretrained_manifest_exists():
    manifests = list_pretrained_models()
    ids = {m["model_id"] for m in manifests}
    assert DEFAULT_PRETRAINED_MODEL_ID in ids


def test_load_default_pretrained_manifest():
    manifest = load_pretrained_manifest(DEFAULT_PRETRAINED_MODEL_ID)
    assert manifest["model_id"] == DEFAULT_PRETRAINED_MODEL_ID
    assert manifest["checkpoint_filename"].endswith(".pt")
    assert len(manifest["sha256"]) == 64
