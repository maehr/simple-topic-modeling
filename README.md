# Browser Topic Explorer

Browser Topic Explorer finds themes in a collection of documents. It runs fully in your browser.

Your documents stay on your computer. The app sends no text to a server.

## What it does

1. You add a corpus. Drag files in, paste text, or load the demo data.
2. You choose a language and edit the stop words.
3. You fit a topic model. The app uses TF-IDF and NMF by default.
4. You explore the topics. Read the top terms, the charts, and the example documents.
5. You rename a topic. The app does not refit the model.
6. You export the results as CSV, JSON, or ZIP.

The app reads CSV, TSV, JSON, JSONL, and plain-text files. Plain text includes Markdown, HTML, XML,
and any other UTF-8 file.

## Run it

Install the dependencies:

```bash
uv sync
```

Open the notebook:

```bash
uv run marimo edit app.py
```

## Build the static app

Build the package wheel first. The browser installs the app from this file.

```bash
uv build --wheel -o public/wheels
```

Check the browser compatibility:

```bash
uv run marimo check app.py --select MW
```

Export the app:

```bash
uv run marimo export html-wasm app.py -o dist --mode run
```

Serve the output over HTTP. The app does not work from a `file://` address.

```bash
python -m http.server --directory dist 8000
```

## Develop

Run the full gate before each commit:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest --doctest-modules --cov --cov-fail-under=100
```

`app.py` holds the user interface. `browser_topics/` holds the logic. Tests cover the logic.

Check the notebook after you edit it. `marimo check` does not catch a name that two cells both
define.

```bash
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('notebook_app', 'app.py')
module = importlib.util.module_from_spec(spec); sys.modules['notebook_app'] = module
spec.loader.exec_module(module); module.app.run(); print('cells ok')
"
```

## License

AGPL-3.0-only. See `LICENSE`.

The bundled stop-word lists come from other projects. See `NOTICE`.
