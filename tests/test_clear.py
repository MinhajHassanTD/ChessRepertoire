from __future__ import annotations

from src.clear import clear_runs


def test_clear_runs_deletes_only_pkl_by_default(tmp_path):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir()
    keep_file = runs_dir / 'notes.txt'
    delete_file = runs_dir / 'STATIC_l1.0_s1000.pkl'
    keep_file.write_text('keep me', encoding='utf-8')
    delete_file.write_bytes(b'data')

    deleted = clear_runs(str(runs_dir))

    assert delete_file in deleted
    assert not delete_file.exists()
    assert keep_file.exists()


def test_clear_runs_delete_all_files_when_requested(tmp_path):
    runs_dir = tmp_path / 'runs'
    nested_dir = runs_dir / 'nested'
    nested_dir.mkdir(parents=True)

    pkl_file = runs_dir / 'COEVOLVE_l1.0_s1000.pkl'
    txt_file = nested_dir / 'meta.txt'
    pkl_file.write_bytes(b'data')
    txt_file.write_text('remove me', encoding='utf-8')

    deleted = clear_runs(str(runs_dir), remove_all_files=True)

    assert pkl_file in deleted
    assert txt_file in deleted
    assert not pkl_file.exists()
    assert not txt_file.exists()
