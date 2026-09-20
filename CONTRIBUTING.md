# Contributing

Thank you for your interest in Browser Topic Explorer.

Please discuss a large change in an [issue](https://github.com/maehr/simple-topic-modeling/issues)
before you write the code. A small fix needs no issue.

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Please follow it in every
interaction with the project.

`AGENTS.md` holds the full rules for this repository. Read it before you change the code.

## Set up

Install [uv](https://docs.astral.sh/uv/) first. Then install the dependencies:

```bash
uv sync
```

## Run the gate

The gate must pass before each commit. The same commands run in CI, so CI gives no surprise.

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest --doctest-modules --cov --cov-fail-under=100
```

Coverage on `browser_topics/` must stay at 100%. Keep each exclusion narrow and explicit.

Install the hooks to run the gate automatically:

```bash
uv run prek install
```

## Check both targets

The project has two targets. A change can pass one and fail the other. Check both.

The notebook runs local CPython:

```bash
uv run marimo edit app.py
```

Check the cell graph after you edit `app.py`. `marimo check` does not find a name that two cells
both define.

```bash
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('notebook_app', 'app.py')
module = importlib.util.module_from_spec(spec); sys.modules['notebook_app'] = module
spec.loader.exec_module(module); module.app.run(); print('cells ok')
"
```

The export runs Pyodide in the browser:

```bash
uv build --wheel -o public/wheels
uv run marimo export html-wasm app.py -o dist --mode run
python -m http.server --directory dist 8000
```

Open `http://localhost:8000`. The app cannot start from a `file://` address.

## Write the commit message

Write each commit message as a [Conventional Commit](https://www.conventionalcommits.org/). The
type decides the changelog group. Write the message by hand; this repository has no `commitizen`
hook.

```text
feat: add the topic map
fix: keep the last result after a failed fit
docs: explain the stop-word lists
```

## Open the pull request

1. Fork the repository and create a branch.
2. Run the gate and both target checks.
3. Title the pull request as a Conventional Commit.
4. Put one logical change in one pull request.

The maintainer reviews and merges. A pull request needs no second approval, because this project
has one maintainer.
