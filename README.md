# Browser Topic Explorer

Browser Topic Explorer finds the themes in a collection of documents. It runs fully in your browser.

The app is a [marimo](https://marimo.io/) notebook. The export runs it with
[Pyodide](https://pyodide.org/), which is Python compiled to WebAssembly. A static web server is
enough to host it. No Python server is needed.

## Privacy

Your documents stay on your computer. The app sends no text to an analysis server. The app calls no
model API. The app builds every download in your browser.

## What it does

1. Add a corpus. Drag files in, paste text, or load the demo data.
2. Choose a language. Edit the stop words when you need to.
3. Choose a model and its parameters.
4. Select **Run model**. The app fits the model in your browser.
5. Explore the topics in the five tabs.
6. Rename a topic. The app does not refit the model.
7. Download the results.

The app fits a model only when you select **Run model**. A changed setting does not start a new fit.
The app keeps the last result until a new fit succeeds.

## What it reads

The app reads two kinds of file.

Use a structured file when one row holds one document plus its metadata:

| Format | Extension |
|---|---|
| Comma-separated values | `.csv` |
| Tab-separated values | `.tsv`, `.tab` |
| JSON | `.json` |
| JSON Lines | `.jsonl`, `.ndjson` |

Use a text file when the file itself is one document. The app reads any UTF-8 text file. It removes
the tags from HTML and XML. It removes the common syntax from Markdown.

The app rejects a binary file, such as a PDF file, a DOCX file, an image, or a ZIP archive.

## Languages

The app ships a stop-word list for each language below. The language changes the stop words only. It
adds no stemming and no grammar analysis.

| Language | Code |
|---|---|
| English | `en` |
| German | `de` |
| French | `fr` |
| Italian | `it` |
| Spanish | `es` |

A corpus with more than one language can turn the base list off and add its own words.

## Models

| Model | When to use it |
|---|---|
| NMF, with TF-IDF | Start here. It is fast, and it suits a short or medium document. |
| LDA, with counts | Use it for a probabilistic topic mixture, or for a long document. |

## What you get

The explorer holds five tabs:

- **Overview** shows the topic cards, the topic map, the prevalence bars, and the similarity heatmap.
- **Topics** shows the top terms, the word cloud, and the representative documents of one topic.
- **Documents** shows the document map, a text search, a topic filter, and a score filter.
- **Metadata** shows the topic mix per group and the topic share over time. It needs a group column
  or a date column.
- **Diagnostics** shows the descriptive numbers and the friendly notices. These numbers are not a
  quality score.

You can download six files:

| File | Content |
|---|---|
| `documents_topics.csv` | One row per document, with its topic scores |
| `topics.csv` | One row per topic, with its name and prevalence |
| `topic_terms.csv` | One row per topic and term, in long format |
| `topic_similarity.csv` | The cosine similarity of each topic pair |
| `config.json` | The settings that produced the result |
| `project.zip` | The five files above, plus a `README.txt` |

## Install

Install [uv](https://docs.astral.sh/uv/) first. Then install the dependencies:

```bash
uv sync
```

## Run the notebook

```bash
uv run marimo edit app.py
```

This command runs local CPython. It does not test the browser build.

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

Serve the output over HTTP:

```bash
python -m http.server --directory dist 8000
```

Then open `http://localhost:8000`.

**Caution:** the app cannot start from a `file://` address. The browser blocks the worker that
Pyodide needs.

## Layout

```text
.
├── app.py                  the user interface, the state, and the downloads
├── browser_topics/         the logic, which the tests cover
│   ├── config.py           the settings models
│   ├── errors.py           one friendly message per failure
│   ├── exports.py          the CSV, JSON, and ZIP files
│   ├── io.py               the import, the split, and the corpus statistics
│   ├── metrics.py          the similarity, the diversity, and the notices
│   ├── modeling.py         the vectorizer and the model fit
│   ├── plots.py            the chart data and the chart specifications
│   ├── preprocess.py       the text cleaning
│   ├── result.py           the result object
│   ├── stopwords.py        the packaged stop-word lists
│   └── data/               the stop-word lists and the demo corpus
├── public/wheels/          the built wheel, which the export copies
└── tests/
```

`app.py` holds the user interface only. `browser_topics/` holds the logic. The tests cover the
logic, not the notebook.

## Develop

Run the full gate before each commit:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest --doctest-modules --cov --cov-fail-under=100
```

Check the notebook after you edit it. `marimo check` does not find a name that two cells both define.

```bash
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('notebook_app', 'app.py')
module = importlib.util.module_from_spec(spec); sys.modules['notebook_app'] = module
spec.loader.exec_module(module); module.app.run(); print('cells ok')
"
```

Build the export and open it in a browser before you call the work done. The notebook and the export
are two targets. A change can pass one and fail the other.

Write each commit message as a [Conventional Commit](https://www.conventionalcommits.org/). Then
update the changelog:

```bash
git-cliff --tag v0.1.0 -o CHANGELOG.md
```

`AGENTS.md` holds the rules for this repository. Read it before you change the code.

## Documents in this repository

| File | Content |
|---|---|
| `AGENTS.md` | The rules for an agent or a contributor |
| `SPECS.md` | The product specification |
| `CHANGELOG.md` | The record of every notable change |
| `NOTICE` | The origin and the licence of the stop-word lists |
| `LICENSE` | The full licence text |

## License

AGPL-3.0-only. See `LICENSE`.

The stop-word lists come from [spaCy](https://spacy.io/) under the MIT licence. See `NOTICE`.
