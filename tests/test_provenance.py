from equity_research.provenance import hash_matches, sha256


def test_old_source_can_be_verified_in_explicit_snapshot(tmp_path):
    live = tmp_path / "scripts/old.py"
    old = tmp_path / "snapshot/scripts/old.py"
    old.parent.mkdir(parents=True)
    old.write_text("original")
    expected = sha256(old)
    assert hash_matches(live, expected, root=tmp_path, snapshot=tmp_path / "snapshot")
    old.write_text("modified")
    assert not hash_matches(live, expected, root=tmp_path, snapshot=tmp_path / "snapshot")


def test_no_snapshot_fallback_for_data_or_current_source(tmp_path):
    for name in ["data/input.csv", "outputs/model.json", "config/plan.json", "src/model.py"]:
        live = tmp_path / name
        old = tmp_path / "snapshot" / name
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("original")
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text("modified")
        assert not hash_matches(live, sha256(old), root=tmp_path, snapshot=tmp_path / "snapshot")


def test_current_source_is_checked_against_current_manifest(tmp_path):
    live = tmp_path / "tests/current.py"
    live.parent.mkdir()
    live.write_text("current")
    assert hash_matches(live, sha256(live), root=tmp_path, snapshot=tmp_path / "absent")
