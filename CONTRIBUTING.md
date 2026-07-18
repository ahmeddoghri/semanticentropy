# Contributing

Thanks for taking a look.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Before opening a pull request

- Keep changes focused. One logical change per PR, not a drive-by rewrite.
- Add or update tests for any behaviour you change. CI runs `pytest` on
  Python 3.9, 3.11, and 3.13, plus the example and benchmark, so it will
  find you.
- Run `ruff check .` and `pytest -q` locally before you push.
- If you touch the equivalence check in `cluster.py`, add a labeled group to
  `corpus.py` that would have broken under the old logic, and make sure the
  benchmark accuracy does not regress.
- If you wire in a real NLI model as the equivalence check, keep the
  `equivalent(a, b) -> bool` signature so the clustering and entropy code do
  not need to change.

## Reporting bugs

Open an issue with a minimal reproduction, the expected versus actual result,
and your Python version. For security issues see [SECURITY.md](SECURITY.md).
