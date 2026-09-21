# Simple Topic Modeling

Simple Topic Modeling finds the themes in a collection of documents. It runs fully in your browser.

[![GitHub issues](https://img.shields.io/github/issues/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/issues)
[![GitHub forks](https://img.shields.io/github/forks/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/network)
[![GitHub stars](https://img.shields.io/github/stars/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/stargazers)
[![GitHub license](https://img.shields.io/github/license/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/blob/main/LICENSE)

The app is a [marimo](https://marimo.io/) notebook. The export runs it with
[Pyodide](https://pyodide.org/), which is Python compiled to WebAssembly. A static web server is
enough to host it. No Python server is needed.

## Use it

Open the app at **<https://maehr.github.io/simple-topic-modeling/>**. Nothing to install.

## Version 2 replaces version 1

Version 1 was a Streamlit app that ran through stlite. Version 2 shares no code with it. The app is
now a marimo notebook, it draws every chart with Altair, and it ships one wheel that `hatchling`
builds.

Version 1 receives no further work. Its code stays at the tag
[`v1.0.0`](https://github.com/maehr/simple-topic-modeling/releases/tag/v1.0.0) and the branch
[`legacy/streamlit`](https://github.com/maehr/simple-topic-modeling/tree/legacy/streamlit).

## The demo corpus

The app opens with a demo corpus, so you can see a real result before you load your own documents.

The corpus holds 295 articles from the *Journal de Genève* and the *Gazette de Lausanne* of 1914.
Each article carries its publication date and the section heading that the newspaper printed above
it. Six sections give six clear themes: the military chronicle, sport, finance, the weather, book
reviews, and the courts. A machine read the articles from a scan, so some words carry errors. A
real archive looks like this.

The Digital Humanities Laboratory of the EPFL digitised the historical archive of
[*Le Temps*](https://www.letempsarchives.ch/). It published the year 1914 under CC BY 4.0, for the
2015 [Swiss Open Cultural Data Hackathon](https://hack.glam.opendata.ch/project/234). The articles
are anonymous newspaper text from 1914, so they left copyright in 1985. [`NOTICE`](NOTICE) holds
the full statement.

`scripts/build_demo_corpus.py` rebuilds the corpus from the archive. It checks the archive against
a known hash and records each source URL and each SHA-256 in
`scripts/demo_corpus.provenance.json`.

## Repeat a run

Every run uses a fixed random seed. The same settings on the same corpus give the same topics.

1. Download `config.json` in Step 4. It records the app version, the language, the stop words, and
   every model parameter, including the seed.
2. Download the demo corpus in Step 1, or keep your own corpus.
3. Send both files to your reader.
4. The reader loads `config.json` in Step 2. The app restores each setting.

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

Step 2 shows the model and the number of topics. The other parameters wait in a closed **Advanced
settings** panel.

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
├── simple_topic_modeling/  the logic, which the tests cover
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
├── scripts/                the script that builds the demo corpus
└── tests/
```

`app.py` holds the user interface only. `simple_topic_modeling/` holds the logic. The tests cover the
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
git-cliff --tag v2.0.0-alpha -o CHANGELOG.md
```

`AGENTS.md` holds the rules for this repository. Read it before you change the code.

## Documents in this repository

| File | Content |
|---|---|
| `AGENTS.md` | The rules for an agent or a contributor |
| `SPECS.md` | The product specification |
| `CHANGELOG.md` | The record of every notable change |
| `CONTRIBUTING.md` | How to set up, to check, and to open a pull request |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 3.0 |
| `SECURITY.md` | How to report a security issue |
| `NOTICE` | The origin and the licence of the demo corpus and the stop-word lists |
| `LICENSE` | The full licence text |

## Author

[Moritz Mähr](https://github.com/maehr) wrote and maintains Simple Topic Modeling.

## Take part

- [Report a problem or ask for a feature](https://github.com/maehr/simple-topic-modeling/issues).
- Read the [contribution guidelines](CONTRIBUTING.md) before you open a pull request.
- Read the [code of conduct](CODE_OF_CONDUCT.md).
- Read [how to report a security issue](SECURITY.md).

## License

AGPL-3.0-only. See `LICENSE`.

The stop-word lists come from [spaCy](https://spacy.io/) under the MIT licence. The demo corpus
comes from the historical archive of *Le Temps*, under CC BY 4.0. See `NOTICE`.
