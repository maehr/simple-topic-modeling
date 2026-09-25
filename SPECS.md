# SPECS.md — Simple Topic Modeling

**Status:** Draft implementation specification  
**Target:** marimo app exported as HTML/WebAssembly and run with Pyodide  
**Goal:** A simple, privacy-friendly browser tool for topic modelling, with strong visualization and easy export.

## 1. Product

Simple Topic Modeling lets users upload or paste a corpus, adjust preprocessing and stop words, fit a topic model locally in the browser, explore topics visually, rename them, and export the results.

The app should feel like a focused web tool rather than a notebook. Code is hidden in run mode.

### Principles

- **Local first:** corpus text stays in the browser.
- **Visualization first:** charts and linked exploration are the main result.
- **Simple by default:** advanced settings are hidden until needed.
- **Explicit fitting:** changing a parameter does not automatically rerun the model.
- **Exportable:** every important result can be downloaded as standard data.
- **Explain jargon:** every model and non-obvious parameter has friendly guidance.

> **Info box — Privacy**  
> Your documents are processed locally in this browser. They are not sent to an analysis server.

---

## 2. User flow

### Onboarding

The app opens for a reader who has never fitted a topic model. The first screen must answer four
questions before any control appears: what the app finds, what the reader gets, how long a run
takes, and what to do first.

Requirements:

- Show an intro above Step 1. Name the app. State what it finds. State what the reader gets. State
  the length of a run.
- Show a bold line that says that a demo corpus is loaded. Point it at **Run model** in Step 3.
- Keep the privacy callout under the intro.
- Show two closed accordions. **How to use this tool** holds the four steps and describes the demo
  corpus. **What is a topic model?** explains a topic, a term, and a score without jargon.
- Show an orientation line above the result tabs. Name what each tab holds. Name the next action.
- Show a **Glossary** accordion under the result tabs. Define topic, term, prevalence, dominant
  score, topic diversity, document frequency, n-gram, and TF-IDF.
- Point each empty state at the control that fills it.
- Show an **About** section at the end. Name the author. Describe and link the demo corpus. Link
  the repository, the issue tracker, and the contribution guidelines.

### Step 1 — Add data

Support drag-and-drop, file picker, paste, and demo data.

The demo corpus holds 295 articles from the *Journal de Genève* and the *Gazette de Lausanne* of
1914, with a `date` column and a `category` column. The EPFL Digital Humanities Laboratory
published the archive under CC BY 4.0 for the 2015 Swiss Open Cultural Data Hackathon. The corpus
ships inside the wheel at `simple_topic_modeling/data/demo_corpus.csv`, so the app reads it without
a network fetch. `scripts/build_demo_corpus.py` rebuilds it and records the provenance. `NOTICE`
holds the source and the licence.

#### Accepted files

Accept **one or multiple browser-readable text files**, not only CSV.

Structured formats with dedicated import behavior:

- `.csv`
- `.tsv`
- `.json`
- `.jsonl` / `.ndjson`

Plain-text or markup files should use the generic text importer, including for example:

- `.txt`, `.text`
- `.md`, `.markdown`
- `.rst`
- `.log`
- `.tex`
- `.html`, `.htm`
- `.xml`
- `.yaml`, `.yml`
- other UTF-8 text files regardless of extension

Binary files such as PDF, DOCX, images, audio, and ZIP are out of scope for MVP.

For unknown extensions, attempt UTF-8 decoding. If successful, treat the file as text; otherwise show a clear unsupported/binary-file message.

> **Info box — Which format works best?**  
> Use CSV/TSV/JSON when you have one row or record per document plus metadata. Use TXT/Markdown or other text files when each file itself is a document.

#### Import behavior

For CSV/TSV/JSON:

- choose the text field/column,
- optional document ID,
- optional date,
- optional group/category fields.

For multiple plain-text files:

- default: **one file = one document**,
- document ID defaults to filename.

For one plain-text file, offer:

- whole file = one document,
- split on blank lines,
- split by line.

For HTML/XML, strip tags before modelling. For Markdown and other markup, keep readable text and remove obvious syntax where practical without introducing heavy parser dependencies.
Keep the line breaks, so the split still finds the paragraphs.

#### Long-document mode

A book, a thesis, a transcript, or a report is one long text. A topic model needs many
observations, so the app splits the text into ordered segments. Each segment is one modelling
document. The model stays a topic model over many segments. It never fits one unsplit text.

When exactly one text is loaded, show **Analyse as**:

- **Corpus document** — the default. The segments are unrelated documents. The app adds no
  position metadata.
- **Long document** — the segments keep their parent and their position.

Both modes use the same split control and the same `split_text()` function. The default split is
on blank lines, which gives one segment per paragraph.

The split must:

- keep the source order,
- drop the empty segments before it numbers them,
- give each segment a stable, 1-based ID: `book.txt#1`, `book.txt#2`, and so on.

In long-document mode, each segment carries three metadata columns:

| Column | Meaning |
|---|---|
| `parent_document_id` | the name of the source text |
| `segment_index` | the 0-based position in the source |
| `segment_number` | the 1-based position in the source |

The order lives in these columns, not only in the ID. A sample keeps the order.

Show:

- document count,
- missing/empty count,
- duplicate count,
- median text length,
- preview table.

Empty documents are excluded with the count shown.

### Step 2 — Configure

Configuration has three sections: **Language & cleaning**, **Model**, **Run**.

### Step 3 — Explore

Tabs:

1. **Overview**
2. **Topics**
3. **Documents**
4. **Metadata** — only when group/date metadata exists
5. **Diagnostics**

Topic selection is shared across tabs.

### Step 4 — Export

Users can download individual CSV/JSON files or one ZIP bundle.

---

## 3. Language and preprocessing

### Language selector

Supported presets:

| UI label | Code | Base stop-word list |
|---|---|---|
| English | `en` | English |
| German | `de` | German |
| French | `fr` | French |
| Italian | `it` | Italian |
| Spanish | `es` | Spanish |

Accept `sp` as an alias for `es` in imported configuration, but always export `es`.

Package the stop-word lists with the app, e.g.:

```text
simple_topic_modeling/data/stopwords/en.txt
simple_topic_modeling/data/stopwords/de.txt
simple_topic_modeling/data/stopwords/fr.txt
simple_topic_modeling/data/stopwords/it.txt
simple_topic_modeling/data/stopwords/es.txt
```

Do not depend on downloading stop-word lists at runtime.

### Recommended defaults

Use the same safe baseline for all five languages, with the language-specific stop-word list selected automatically.

| Setting | EN | DE | FR | IT | ES |
|---|---:|---:|---:|---:|---:|
| lowercase | on | on | on | on | on |
| strip URLs | on | on | on | on | on |
| strip emails | on | on | on | on | on |
| strip numbers | off | off | off | off | off |
| accent normalization | off | off | off | off | off |
| minimum token length | 2 | 2 | 2 | 2 | 2 |
| base stop words | EN | DE | FR | IT | ES |
| n-grams | 1 | 1 | 1 | 1 | 1 |

Language affects stop words only in MVP; do not imply stemming, lemmatization, or grammatical parsing.

> **Info box — Language**  
> Choose the main language of your documents. This mainly changes common words that are ignored, such as “the”, “und”, “le”, “di”, or “el”. Mixed-language corpora can use **No base stop words** plus a custom list.

### Stop-word editor

Controls:

- base list: selected language / none,
- **Add stop words** textarea,
- **Always keep these words** textarea,
- effective stop-word count,
- searchable preview,
- reset button.

Accept comma-, whitespace-, or newline-separated input.

Effective set:

```text
(base_stop_words UNION added_stop_words) MINUS always_keep_words
```

> **Info box — Stop words**  
> Add words that occur often but are not meaningful for your question, such as company names, boilerplate, months, or survey wording. Avoid removing words just because they are frequent; some frequent words may define an important topic.

Show a cheap **frequent terms preview** after preprocessing. This preview must not fit the topic model.

---

## 4. Models and parameters

### Model selector

#### NMF — default and recommended

Use TF-IDF + Non-negative Matrix Factorization.

> **When to use NMF**  
> Start here for most collections. It is fast, works well with short and medium-length documents, and usually produces easy-to-read keyword topics. Its topic scores are relative weights, not probabilities.

Recommended implementation:

```python
TfidfVectorizer(
    stop_words=effective_stop_words,
    min_df=min_df,
    max_df=max_df,
    max_features=max_features,
    ngram_range=ngram_range,
    lowercase=False,
    sublinear_tf=True,
)

NMF(
    n_components=n_topics,
    init="nndsvda",
    max_iter=max_iter,
    random_state=42,
)
```

Normalize each document's NMF topic weights to sum to 1 for display/export and call them `topic_score` or `topic_share`.

#### LDA — optional

Use `CountVectorizer` + scikit-learn `LatentDirichletAllocation`.

> **When to use LDA**  
> Try LDA when you specifically want a probabilistic topic mixture or have medium/long documents. It may be slower and more sensitive to parameter choices than NMF. If unsure, use NMF.

### Basic parameters

Defaults:

| Parameter | Default | Range / choices |
|---|---:|---|
| Model | NMF | NMF / LDA |
| Number of topics | 10 | 2–30 |
| Max vocabulary | 5,000 | 2,000 / 5,000 / 10,000 / custom ≤20,000 |
| Minimum document frequency | 2 | integer ≥1 |
| Maximum document frequency | 0.95 | 0.5–1.0 |
| N-grams | unigrams | 1 / 1–2 / 2 |
| Random seed | 42 | advanced |
| Max iterations | model-specific sensible default | advanced |

### Friendly parameter help

Show these descriptions inline or in tooltips.

> **Number of topics**  
> How many themes the model should look for. Start around 8–12. Increase it if topics are too broad; decrease it if many topics look almost identical or tiny.

> **Max vocabulary**  
> Maximum number of distinct terms considered. 5,000 is a good browser-friendly default. Increase it for large, varied corpora; lower it for speed.

> **Minimum document frequency (`min_df`)**  
> Ignore words that occur in fewer than this many documents. Raise it to remove typos, names, or very rare terms. Lower it for small corpora.

> **Maximum document frequency (`max_df`)**  
> Ignore words that occur in almost every document. `0.95` means words appearing in more than 95% of documents are ignored.

> **N-grams**  
> Unigrams use single words. `1–2` also includes two-word phrases such as “climate change”. Phrases can improve labels but make the vocabulary larger and slower.

> **Iterations**  
> Usually leave this alone. Increase it only if the app reports that the model did not converge.

Include **Reset recommended defaults**.

### Run behavior

Show a summary before fitting, e.g.:

> 2,418 documents · English · NMF · 10 topics · max 5,000 terms

Primary button: **Run model**.

Changing settings after a successful run shows:

> **Configuration changed — rerun to update results.**

Keep the previous results visible until a new run succeeds.

---

## 5. Result object

Charts and exports consume one model-agnostic result object.

```python
TopicModelResult = {
    "model_type": str,
    "config": dict,
    "documents": list[str],
    "document_ids": list[str],
    "metadata": pandas.DataFrame,
    "feature_names": list[str],
    "document_topic_raw": numpy.ndarray,
    "document_topic": numpy.ndarray,
    "topic_term_raw": numpy.ndarray,
    "topic_term": numpy.ndarray,
    "document_xy": numpy.ndarray,
    "topic_xy": numpy.ndarray,
    "topic_names": list[str],
    "topic_auto_labels": list[str],
    "topic_prevalence": numpy.ndarray,
    "dominant_topic": numpy.ndarray,
    "dominant_topic_score": numpy.ndarray,
    "metrics": dict,
}
```

Initial topic labels use the three strongest terms:

```text
Topic 1 · economy, market, growth
```

Users can rename topics without refitting.

### 2-D projection

Use `TruncatedSVD(n_components=2, random_state=42)` on the document-term matrix.

Topic coordinates are weighted centroids of document coordinates using document-topic scores.

Axes are for visual separation only and should be visually de-emphasized.

---

## 6. Visualizations

Use **Altair/Vega-Lite** by default. Keep plotting functions isolated so the chart layer can be changed later.

Every chart needs a title, hover tooltip, readable labels, and an empty state.

### Overview

- topic cards,
- topic map,
- topic prevalence bars,
- topic similarity heatmap.

**Topic card:** name, top five terms, prevalence, click to select.

**Topic map:** bubble position = 2-D centroid, size = prevalence, click = select topic.

**Similarity heatmap:** cosine similarity between normalized topic-term vectors.

### Topics

For the selected topic:

- editable topic name,
- top-term horizontal bar chart,
- word cloud,
- prevalence,
- representative documents.

Representative documents are ranked by the selected topic score and show ID, score, snippet, and selected metadata.

In long-document mode, the tab shows two more items:

- **Where this topic occurs** — an area chart of the topic share along the segment number.
- **Representative passages** — the table replaces the representative documents. Each row shows
  the segment number, the dominant topic, the dominant score, the score of the selected topic,
  and a snippet.

> **Info box — How to read a topic**  
> Look at the top terms together with several high-scoring documents. Topic keywords are clues, not a complete definition. Rename the topic once its meaning is clear to you.

### Documents

- 2-D document scatter plot,
- color by dominant topic,
- hover with ID, topic, score, snippet,
- linked searchable/filterable table,
- topic filter,
- minimum score filter,
- metadata filters when available.

In long-document mode, the tab opens with a **position heatmap**:

- x-axis: the segment number, in source order,
- y-axis: the topics, in topic order,
- colour: the topic share, on one sequential blue scale.

Above 300 segments, the heatmap averages neighbouring segments into 300 columns. The tooltip
names the first and the last segment of each column. The table keeps the source order and shows
the segment number, the dominant topic, and its score.

### Metadata

If a group column exists:

- mean topic share by group,
- normalized stacked bar chart.

If a date column exists:

- parse dates with parsed/unparsed count,
- choose sensible day/month/year bins,
- topic share over time line chart.

### Diagnostics

Show descriptive aids only:

- documents used,
- vocabulary size,
- topic count,
- topic diversity,
- mean pairwise topic similarity,
- dominant-topic score distribution,
- NMF reconstruction error or LDA perplexity when available.

Friendly warnings may say:

- “Several topics use very similar terms — consider fewer topics.”
- “Many documents have weak dominant-topic scores — inspect whether the corpus contains mixed themes.”
- “The vocabulary reached the configured maximum — increase it if important terms appear to be missing.”

Do not present diagnostics as an automatic quality score.

---

## 7. Export

### `documents_topics.csv`

One row per modelled document:

```text
document_id
text                       # optional toggle
<metadata columns>         # parent_document_id, segment_index, segment_number in long-document mode
dominant_topic_id
dominant_topic_name
dominant_topic_score
topic_0_score ... topic_K-1_score
projection_x
projection_y
```

### `topics.csv`

```text
topic_id
topic_name
topic_auto_label
prevalence
x
y
top_terms
```

### `topic_terms.csv`

Long format:

```text
topic_id
topic_name
rank
term
weight_raw
weight_normalized
```

### `topic_similarity.csv`

```text
topic_a_id
topic_a_name
topic_b_id
topic_b_name
cosine_similarity
```

### `config.json`

Include:

- app version,
- language code,
- model and parameters,
- preprocessing settings,
- base/add/keep stop-word settings,
- random seed,
- `analyse_as` (`corpus` or `long_document`),
- `split_mode` (`whole`, `blank_lines`, `lines`, or `null` when the app did not split the input).

A `config.json` without `analyse_as` or `split_mode` still loads. The two fields take their
defaults: `corpus` and `null`.

### `project.zip`

```text
documents_topics.csv
topics.csv
topic_terms.csv
topic_similarity.csv
config.json
README.txt
```

Create the ZIP locally with Python's standard `zipfile` module.

Optionally expose Vega-Lite chart specs as JSON. PNG/SVG chart download is nice-to-have, not an MVP blocker.

### Reproducibility

A reader must be able to repeat a run exactly. Two files carry everything that a run needs: the
corpus and `config.json`.

Requirements:

- Fix the random seed. Store it in `config.json`. Show it in the advanced settings, so a reader can
  change it and see how stable the topics are.
- Stamp `app_version` from the installed package metadata, never from a literal. A file from
  version 2 must report version 2.
- Offer a download of the demo corpus while the demo is the active source.
- Accept a `config.json` upload. Restore the language, the stop words, every model parameter, and
  the split settings of one text. Name the loaded file and the version that wrote it.
- Report a bad file with a recovery action. Keep the parsing in the package, never in the notebook,
  so the tests can cover it.
- Let **Reset recommended defaults** discard a loaded configuration.

---

## 8. Browser performance and errors

Recommended target:

- ideal: up to ~5,000 short/medium documents,
- target: up to ~10,000 documents on a modern laptop,
- warn above ~10,000 documents or ~50 MB of text,
- offer sampling for very large inputs.

Performance rules:

- keep matrices sparse,
- do not densify the full document-term matrix,
- fit only on **Run model**,
- cache derived coordinates/similarities per run,
- use snippets rather than full text in tables,
- sample scatter-plot points if rendering becomes slow without changing model results.

Required friendly errors:

- unsupported/binary file,
- text cannot be decoded,
- no usable text,
- too few documents,
- empty vocabulary,
- too many topics for the data,
- model did not converge,
- date parsing failed,
- browser memory/computation failure.

Errors should suggest a recovery action.

Example:

> No terms remain after filtering. Try lowering **Minimum document frequency** or removing some stop words.

A failed rerun must not erase the last successful result.

---

## 9. marimo architecture

Suggested layout:

```text
.
├── app.py
├── pyproject.toml
├── SPECS.md
├── simple_topic_modeling/
│   ├── io.py
│   ├── preprocess.py
│   ├── modeling.py
│   ├── result.py
│   ├── metrics.py
│   ├── plots.py
│   ├── exports.py
│   ├── stopwords.py
│   └── data/
│       ├── demo_corpus.csv
│       └── stopwords/
│           ├── en.txt
│           ├── de.txt
│           ├── fr.txt
│           ├── it.txt
│           └── es.txt
└── tests/
```

`app.py` handles UI, state, explicit run trigger, rendering, and downloads. Core modelling/import/export logic lives in testable modules.

### Reactivity

```text
upload -> parsed corpus -> preprocessing preview
settings -> pending config
Run button + corpus + config -> TopicModelResult   [expensive]
TopicModelResult + explore state -> charts/tables  [cheap]
TopicModelResult -> exports                         [on demand]
```

The expensive modelling cell must depend on the explicit run action, not directly refit on every widget change.

### Dependencies

Keep the runtime small and Pyodide-friendly:

```text
marimo
numpy
pandas
scipy
scikit-learn
altair
wordcloud
Pillow
```

---

## 10. Build and privacy

Development:

```bash
marimo edit app.py
```

Check browser compatibility:

```bash
marimo check app.py --select MW
```

Export:

```bash
marimo export html-wasm app.py -o dist --mode run
```

Serve the output over HTTP; do not rely on `file://`.

Privacy requirements:

- no corpus text sent to a backend,
- no model API calls,
- downloads generated locally,
- analytics must never contain document text, filenames, metadata values, stop words, or topic labels unless explicitly added later with consent.

---

## 11. Acceptance criteria

The MVP is done when a user can:

1. Open the static marimo WASM app without a Python server.
2. Upload CSV/TSV/JSON/JSONL or generic UTF-8 text files including Markdown.
3. Upload multiple text files with one-file-per-document behavior.
4. Select EN/DE/FR/IT/ES defaults and edit stop words.
5. Understand the model and main parameters from built-in explanations.
6. Fit NMF locally with an explicit **Run model** action.
7. Optionally fit LDA.
8. Explore topic cards, topic map, term bars, word cloud, representative documents, similarity heatmap, document map, and metadata views.
9. Rename topics without refitting.
10. Search/filter documents.
11. Download document/topic/term/similarity CSVs, configuration JSON, and project ZIP.
12. Keep all corpus processing local to the browser.

---

## 12. First implementation milestone

Build this vertical slice first:

```text
multi-format text upload
-> choose/split documents
-> language preset + stop-word editor
-> TF-IDF + NMF
-> TopicModelResult
-> topic cards
-> top-term bars
-> representative documents
-> CSV exports
-> html-wasm export
```

Then add maps, heatmap, word cloud, metadata views, diagnostics, LDA, and ZIP export.

## References

- TopicWizard: https://github.com/x-tabdeveloping/topicwizard
- marimo WebAssembly HTML export: https://docs.marimo.io/guides/exporting/webassembly_html/
- Pyodide packages: https://pyodide.org/en/stable/usage/packages-in-pyodide.html
