"""Utility to clear experiment run artifacts so experiments can be rerun.

By default this removes only ``*.pkl`` files in ``runs/`` recursively, which
matches the output format produced by ``src.experiments``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def clear_runs(runs_dir: str = 'runs', remove_all_files: bool = False) -> list[Path]:
	"""Delete run artifacts from *runs_dir* and return deleted file paths.

	Args:
		runs_dir: Directory containing experiment run outputs.
		remove_all_files: If True, delete all files in the directory tree.
			If False, delete only ``*.pkl`` files.
	"""
	root = Path(runs_dir)
	if not root.exists() or not root.is_dir():
		return []

	pattern = '*' if remove_all_files else '*.pkl'
	deleted: list[Path] = []

	for file_path in root.rglob(pattern):
		if file_path.is_file():
			file_path.unlink()
			deleted.append(file_path)

	return deleted


def main() -> int:
	parser = argparse.ArgumentParser(
		description='Clear experiment run files so experiments can be rerun from scratch.'
	)
	parser.add_argument(
		'--runs-dir',
		default='runs',
		help='Directory containing run artifacts (default: runs).',
	)
	parser.add_argument(
		'--all-files',
		action='store_true',
		help='Delete all files in the runs directory, not just *.pkl files.',
	)
	args = parser.parse_args()

	deleted = clear_runs(args.runs_dir, remove_all_files=args.all_files)
	if deleted:
		print(f"Deleted {len(deleted)} file(s) from {args.runs_dir}.")
	else:
		print(f"No matching run files found in {args.runs_dir}.")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
