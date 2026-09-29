# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair",
#     "numpy",
#     "pandas",
#     "pillow",
#     "pydantic",
#     "pypdf",
#     "scikit-learn",
#     "scipy",
#     "wordcloud",
#     "simple-topic-modeling @ public/wheels/simple_topic_modeling-2.0.0a0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Simple Topic Modeling")


@app.cell(hide_code=True)
async def _():
    import sys

    import marimo as mo

    # marimo passes the PEP 723 wheel path to micropip as a relative URL, which micropip does not
    # resolve against this page. Install the wheel again from an absolute URL, so the package is
    # present whichever path ran first. micropip skips a package it already installed.
    if sys.platform == "emscripten":
        import micropip

        _wheel = (
            mo.notebook_location()
            / "public"
            / "wheels"
            / "simple_topic_modeling-2.0.0a0-py3-none-any.whl"
        )
        await micropip.install(str(_wheel))

    import pandas as pd

    from simple_topic_modeling import exports, io, metrics, modeling, plots
    from simple_topic_modeling import result as result_mod
    from simple_topic_modeling.config import (
        LANGUAGE_LABELS,
        MODE_LABELS,
        AppConfig,
        ModelConfig,
        PreprocessConfig,
        StopWordConfig,
        config_changed,
    )
    from simple_topic_modeling.errors import FriendlyMessage, TopicError
    from simple_topic_modeling.preprocess import frequent_terms
    from simple_topic_modeling.selection import resolve_topic, selected_topic_from_chart
    from simple_topic_modeling.stopwords import effective_stopwords

    return (
        AppConfig,
        FriendlyMessage,
        LANGUAGE_LABELS,
        MODE_LABELS,
        ModelConfig,
        PreprocessConfig,
        StopWordConfig,
        TopicError,
        config_changed,
        effective_stopwords,
        exports,
        frequent_terms,
        io,
        metrics,
        mo,
        modeling,
        pd,
        plots,
        resolve_topic,
        result_mod,
        selected_topic_from_chart,
    )


@app.cell(hide_code=True)
def _(mo):
    # The run request lets a control other than the run button start one fit.
    # A later Rerun action reuses it.
    get_source_choice, set_source_choice = mo.state("Demo data")
    get_run_request, set_run_request = mo.state(False)
    return get_run_request, get_source_choice, set_run_request, set_source_choice


@app.cell(hide_code=True)
def _(mo, set_run_request, set_source_choice):
    def _run_demo(_value):
        set_source_choice("Demo data")
        set_run_request(True)

    def _use_own_data(_value):
        set_source_choice("Upload files")

    _intro = mo.md(
        """
    # Simple Topic Modeling

    This app finds the themes in a collection of documents, or in one long text such as a book.

    You get a list of topics. Each topic holds the words that occur together, the share of the
    corpus that the topic covers, and the documents that match it. For one long text, the app
    also shows where each topic occurs, from the first paragraph to the last.

    A run takes a few seconds for a few hundred documents.

    Select **Run demo** to fit a model on the demo corpus now.
    Select **Use my own data** to load your documents in Step 1.
    """
    )
    # marimo holds a UI element by a weak reference. A button with a local name only is garbage
    # collected after the cell runs, and marimo then drops its clicks. Give each button a global
    # name.
    run_demo_button = mo.ui.button(label="Run demo", kind="success", on_change=_run_demo)
    own_data_button = mo.ui.button(label="Use my own data", kind="neutral", on_change=_use_own_data)
    mo.vstack([_intro, mo.hstack([run_demo_button, own_data_button], justify="start")])
    return own_data_button, run_demo_button


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            """
        **Your documents stay in this browser.**

        The app reads and models your text locally. It sends nothing to an analysis server.
        """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How to use this tool": mo.md(
                """
            1. **Add data.** Use the demo corpus, or load your own documents. You can also load
               one long text, such as a book, and choose **Analyse as: Ordered text**.
            2. **Configure.** Set the language, then set the number of topics.
            3. **Run and explore.** Select **Run model**. The app fits the model in this browser.
               Then read each topic and name it.
            4. **Export.** Download the results.

            The demo corpus holds 295 French articles from two Swiss newspapers of 1914. A machine
            read the articles from a scan, so some words carry errors. A real archive looks like
            this.
            """
            ),
            "What is a topic model?": mo.md(
                """
            A topic model reads a collection of documents. It finds the groups of words that occur
            together. Each group is a **topic**.

            A **term** is one word in a topic. The model gives each term a weight. The terms with
            the highest weight tell you what the topic is about.

            A **score** says how strongly one document belongs to one topic. One document can hold
            several topics. The highest score names the main topic of that document.

            The model does not know what a topic means. You read the terms and the documents. Then
            you give the topic a name.
            """
            ),
            "Can I analyse one long text?": mo.md(
                """
            Yes. A topic model needs many documents, so the app splits one long text into
            **segments**, usually its paragraphs. Each segment counts as one document.

            In **Ordered text** mode, each segment keeps its number, which is its position in the
            text. The results then show where each topic occurs, and which passages represent it
            best. Use this mode for a book, a thesis, a report, or a transcript.

            Load the text as one TXT, Markdown, HTML, or text-based PDF file, or paste it. One PDF
            opens in this mode. A PDF segment is usually one page, not one paragraph.

            In **Independent segments** mode, the segments are unrelated documents, and their order
            is lost.
            """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 1 — Add data""")
    return


@app.cell(hide_code=True)
def _(get_source_choice, mo, set_source_choice):
    source = mo.ui.radio(
        options=["Demo data", "Upload files", "Paste text"],
        value=get_source_choice(),
        on_change=set_source_choice,
        label="Where does your text come from?",
        inline=True,
    )
    source
    return (source,)


@app.cell(hide_code=True)
def _(mo, source):
    file_input = mo.ui.file(multiple=True, kind="area", label="Drop files here")
    paste_input = mo.ui.text_area(
        label="Paste your text", placeholder="One document per paragraph.", rows=8
    )
    _widget = {"Upload files": file_input, "Paste text": paste_input}.get(source.value)
    _help = mo.accordion(
        {
            "Which format works best?": mo.md(
                """
            Use **CSV, TSV, JSON, or JSONL** when you have one row per document plus metadata
            such as a date or a category.

            Use **TXT, Markdown, HTML, or any other UTF-8 text file** when each file is itself
            one document.

            Use **one long text**, such as a book, a thesis, or a transcript, to see where each
            topic occurs in it. Load the one file, then choose **Analyse as: Ordered text**. The
            app splits the text into paragraphs and keeps their order.

            Use a **text-based PDF** in the same way. The app reads the text in this browser. One
            PDF opens as an ordered text. A PDF rarely holds blank lines, so each page usually
            becomes one segment. A scanned PDF holds images, not text. The app does no OCR, so
            convert a scan to a searchable PDF first. The app does not open a PDF that needs a
            password to open.

            Upload one table, or any number of text and PDF files. The app rejects any other mix.

            DOCX, images, and ZIP files are out of scope.
            """
            )
        }
    )
    mo.vstack([_widget, _help]) if _widget is not None else _help
    return file_input, paste_input


@app.cell(hide_code=True)
def _(exports, io, mo, source):
    # The demo corpus is the one file a reader needs to repeat a demo run. It ships inside the
    # wheel, so the download reads it from the package and never from the network.
    if source.value == "Demo data":
        _demo = mo.vstack(
            [
                mo.download(
                    data=exports.to_csv_bytes(io.demo_table()),
                    filename="demo_corpus.csv",
                    label="Download the demo corpus",
                ),
                mo.md(
                    "*295 articles from the* Journal de Genève *and the* Gazette de Lausanne *of"
                    " 1914. See `NOTICE` for the source and the licence.*"
                ),
            ]
        )
    else:
        _demo = mo.md("")
    _demo
    return


@app.cell(hide_code=True)
def _(TopicError, file_input, io, paste_input, source):
    table = None
    text_documents = None
    text_names = None
    load_error = None
    load_notices = []

    # A PDF that fails raises here, so the message takes the existing load-error path. The fit
    # state stays untouched, so the last result survives a bad PDF.
    try:
        if source.value == "Demo data":
            table = io.demo_table()
        elif source.value == "Upload files":
            _uploads = [io.UploadedFile(item.name, item.contents) for item in file_input.value]
            _plan = io.classify_uploads(_uploads)
            if _plan.error is not None:
                load_error = _plan.error
            elif _plan.table is not None:
                table = io.read_table(_plan.table)
            elif _plan.texts:
                _read = [io.read_text_document(item) for item in _plan.texts]
                text_documents = [text for text, _ in _read]
                text_names = [item.name for item in _plan.texts]
                load_notices = [notice for _, notice in _read if notice is not None]
        elif paste_input.value.strip():
            text_documents = [paste_input.value]
            text_names = ["pasted text"]
    except TopicError as error:
        load_error = error.friendly

    return load_error, load_notices, table, text_documents, text_names


@app.cell(hide_code=True)
def _(MODE_LABELS, get_loaded, mo, text_names):
    # These two controls apply to one text only. A loaded config.json from a run on one text
    # restores them, so a reader splits the text exactly as the author did. Otherwise one PDF
    # opens as an ordered text, and any other text opens as independent segments.
    _loaded = get_loaded()
    _config = _loaded[0] if _loaded is not None else None
    _one_pdf = (
        text_names is not None and len(text_names) == 1 and text_names[0].lower().endswith(".pdf")
    )
    if _config is not None and _config.split_mode is not None:
        _start_mode = _config.analyse_as
    else:
        _start_mode = "long_document" if _one_pdf else "corpus"
    _split_labels = {
        "whole": "Whole file is one document",
        "blank_lines": "Split on blank lines",
        "lines": "Split by line",
    }
    split_mode = mo.ui.dropdown(
        options={label: code for code, label in _split_labels.items()},
        value=_split_labels[
            _config.split_mode
            if _config is not None and _config.split_mode is not None
            else "blank_lines"
        ],
        label="How should the app split this text?",
    )
    analyse_as = mo.ui.radio(
        options={label: code for code, label in MODE_LABELS.items()},
        value=MODE_LABELS[_start_mode],
        label="Analyse as",
        inline=True,
    )
    return analyse_as, split_mode


@app.cell(hide_code=True)
def _(analyse_as, mo, split_mode, table, text_documents, text_names):
    _columns = [str(name) for name in table.columns] if table is not None else []
    _guess = next(
        (name for name in _columns if name.lower() in {"text", "body", "content", "abstract"}),
        _columns[0] if _columns else None,
    )
    text_column = mo.ui.dropdown(options=_columns, value=_guess, label="Text column")
    id_column = mo.ui.dropdown(
        options=["(row number)", *_columns], value="(row number)", label="Document ID"
    )
    _date_guess = next(
        (name for name in _columns if name.lower() in {"date", "published", "created", "year"}),
        "(none)",
    )
    _group_guess = next(
        (
            name
            for name in _columns
            if name.lower() in {"category", "group", "label", "source", "author", "topic"}
        ),
        "(none)",
    )
    date_column = mo.ui.dropdown(
        options=["(none)", *_columns], value=_date_guess, label="Date column"
    )
    group_column = mo.ui.dropdown(
        options=["(none)", *_columns], value=_group_guess, label="Group column"
    )

    if table is not None:
        _controls = mo.hstack(
            [text_column, id_column, date_column, group_column], justify="start", gap=1, wrap=True
        )
    elif text_documents is not None and len(text_documents) == 1:
        _controls = mo.vstack(
            [
                mo.hstack([analyse_as, split_mode], justify="start", gap=2, wrap=True),
                mo.md(
                    "*In **Independent segments** mode, the pieces are unrelated documents. In"
                    " **Ordered text** mode, the pieces keep their order, so the results show"
                    " where each topic occurs in the text.*"
                ),
            ]
        )
    elif text_names:
        _controls = mo.md(f"**{len(text_names)} files.** One file is one document.")
    else:
        _controls = mo.md("*No data yet. Choose **Demo data** above to load the demo corpus.*")
    _controls
    return date_column, group_column, id_column, text_column


@app.cell(hide_code=True)
def _(
    TopicError,
    analyse_as,
    date_column,
    group_column,
    id_column,
    io,
    pd,
    split_mode,
    table,
    text_column,
    text_documents,
    text_names,
):
    corpus = None
    stats = None
    corpus_error = None

    try:
        if table is not None and text_column.value is not None:
            _documents = table[text_column.value].fillna("").astype(str).tolist()
            if id_column.value == "(row number)":
                _identifiers = [str(number + 1) for number in range(len(_documents))]
            else:
                _identifiers = table[id_column.value].astype(str).tolist()
            _extra = {}
            if date_column.value != "(none)":
                _extra["date"] = table[date_column.value]
            if group_column.value != "(none)":
                _extra["group"] = table[group_column.value]
            _metadata = pd.DataFrame(_extra) if _extra else None
            corpus, stats = io.build_corpus(_documents, _identifiers, _metadata)
        elif text_documents is not None and text_names is not None:
            if len(text_documents) == 1:
                # One text always splits the same way. A long document also keeps the parent
                # and the position of each segment as metadata.
                _pieces, _identifiers, _positions = io.split_long_document(
                    text_documents[0], text_names[0], split_mode.value
                )
                _metadata = _positions if analyse_as.value == "long_document" else None
                corpus, stats = io.build_corpus(_pieces, _identifiers, _metadata)
            else:
                corpus, stats = io.build_corpus(text_documents, list(text_names))
    except TopicError as error:
        corpus_error = error.friendly

    return corpus, corpus_error, stats


@app.cell(hide_code=True)
def _(
    analyse_as,
    corpus,
    corpus_error,
    io,
    load_error,
    load_notices,
    mo,
    pd,
    source,
    split_mode,
    stats,
    table,
    text_names,
):
    _problem = load_error or corpus_error
    if _problem is not None:
        _view = mo.callout(mo.md(f"**{_problem.detail}** {_problem.recovery}"), kind="warn")
    elif corpus is None or stats is None:
        _view = mo.md("*No corpus yet.*")
    else:
        _numbers = pd.DataFrame(
            {
                "Documents": [stats.total],
                "Used": [stats.kept],
                "Empty": [stats.empty],
                "Duplicates": [stats.duplicates],
                "Median length": [f"{stats.median_length:.0f} characters"],
            }
        )
        _preview = pd.DataFrame(
            {
                "document_id": corpus.document_ids[:5],
                "text": [document[:160] for document in corpus.documents[:5]],
            }
        )
        # One line says what the model sees. It follows the mode that the reader chose.
        if table is not None:
            _seen = io.describe_analysis(1, len(corpus), "document", "corpus", "table")
        elif text_names is not None and len(text_names) == 1:
            _unit = io.segment_unit(split_mode.value, text_names[0].lower().endswith(".pdf"))
            # Pasted text loads no file, so the line names the paste instead.
            _origin = "pasted text" if source.value == "Paste text" else "file"
            _seen = io.describe_analysis(1, len(corpus), _unit, analyse_as.value, _origin)
        elif text_names is not None:
            _seen = io.describe_analysis(len(text_names), len(corpus), "document", "corpus")
        else:
            _seen = None
        _view = mo.vstack(
            [
                *[
                    mo.callout(mo.md(f"**{_note.detail}** {_note.recovery}"), kind="warn")
                    for _note in load_notices
                ],
                mo.ui.table(_numbers, selection=None),
                mo.md(f"*What the model sees: {_seen}.*") if _seen else mo.md(""),
                mo.md("**Preview**"),
                mo.ui.table(_preview, selection=None),
            ]
        )
    _view
    return


@app.cell(hide_code=True)
def _(corpus, io, mo):
    _warning = io.corpus_size_warning(corpus.documents) if corpus is not None else None
    use_sample = mo.ui.checkbox(value=_warning is not None, label="Model a sample instead")
    # The value must stay inside the bounds, or marimo raises and every later cell fails. A
    # corpus below the minimum shows no sample control, so the clamp changes nothing it models.
    sample_size = mo.ui.number(
        100,
        50_000,
        value=max(100, min(5_000, len(corpus))) if corpus else 5_000,
        step=100,
        label="Sample size",
    )
    _view = (
        mo.vstack(
            [
                mo.callout(
                    mo.md(f"**{_warning.detail}** {_warning.recovery}"),
                    kind="warn",
                ),
                mo.hstack([use_sample, sample_size], justify="start", gap=2),
            ]
        )
        if _warning is not None
        else mo.md("")
    )
    _view
    return sample_size, use_sample


@app.cell(hide_code=True)
def _(corpus, io, sample_size, use_sample):
    modelled_corpus = corpus
    if corpus is not None and use_sample.value:
        modelled_corpus = io.sample_corpus(corpus, int(sample_size.value))
    return (modelled_corpus,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 2 — Configure""")
    return


@app.cell(hide_code=True)
def _(TopicError, config_from_upload, mo, set_loaded, set_touched):
    def _apply(value):
        # marimo calls this when a file arrives, and again with an empty value on a reset.
        if not value:
            return
        _file = value[0]
        try:
            # The loaded file replaces every earlier choice of the reader.
            set_touched({})
            set_loaded((config_from_upload(_file.contents), _file.name, None))
        except TopicError as error:
            set_loaded((None, _file.name, error.friendly))

    config_upload = mo.ui.file(
        filetypes=[".json"],
        multiple=False,
        label="Load settings from a config.json",
        on_change=_apply,
    )
    return (config_upload,)


@app.cell(hide_code=True)
def _(config_upload, get_loaded, mo):
    _loaded = get_loaded()
    if _loaded is None:
        _status = mo.md("")
    elif _loaded[2] is not None:
        _status = mo.callout(mo.md(f"**{_loaded[2].detail}** {_loaded[2].recovery}"), kind="warn")
    else:
        _status = mo.md(
            f"*Loaded the settings from `{_loaded[1]}`, written by version"
            f" {_loaded[0].app_version}. Select **Reset recommended defaults** to discard them.*"
        )
    mo.vstack(
        [
            config_upload,
            _status,
            mo.accordion(
                {
                    "Repeat a run exactly": mo.md(
                        """
                    Every run uses a fixed random seed, so the same settings on the same corpus
                    give the same topics.

                    Download `config.json` in Step 4. Send it with your corpus. The reader loads
                    the file here, and the app restores each setting.
                    """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(LANGUAGE_LABELS, get_loaded, get_reset, mo):
    get_reset()
    _loaded = get_loaded()
    _config = _loaded[0] if _loaded is not None else None
    _stop = _config.stop_words if _config is not None else None
    language = mo.ui.dropdown(
        options={label: code for code, label in LANGUAGE_LABELS.items()},
        value=LANGUAGE_LABELS[_config.language] if _config is not None else "French",
        label="Language",
    )
    use_base = mo.ui.checkbox(
        value=_stop.use_base_list if _stop is not None else True,
        label="Use the base stop-word list",
    )
    added_words = mo.ui.text_area(
        label="Add stop words",
        value=_stop.added if _stop is not None else "",
        placeholder="survey, january, ltd",
        rows=3,
    )
    keep_words = mo.ui.text_area(
        label="Always keep these words",
        value=_stop.always_keep if _stop is not None else "",
        placeholder="growth",
        rows=3,
    )
    _help = mo.accordion(
        {
            "About the language setting": mo.md(
                """
            Choose the main language of your documents. This mainly changes the common words that
            the app ignores, such as *the*, *und*, *le*, *di*, or *el*.

            The demo corpus is French, so the app starts on French. Change this when you load
            your own documents.

            A mixed-language corpus can turn the base list off and add its own words.

            The language does not add stemming or grammar analysis.
            """
            ),
            "About stop words": mo.md(
                """
            Add words that occur often but carry no meaning for your question, such as company
            names, boilerplate, months, or survey wording.

            Do not remove a word only because it is frequent. A frequent word can define an
            important topic.
            """
            ),
        }
    )
    mo.vstack(
        [
            mo.hstack([language, use_base], justify="start", gap=2),
            mo.hstack([added_words, keep_words], justify="start", gap=2, widths="equal"),
            _help,
        ]
    )
    return added_words, keep_words, language, use_base


@app.cell(hide_code=True)
def _(StopWordConfig, added_words, effective_stopwords, keep_words, language, use_base):
    stop_word_config = StopWordConfig(
        use_base_list=use_base.value,
        added=added_words.value,
        always_keep=keep_words.value,
    )
    active_stopwords = effective_stopwords(language.value, stop_word_config)
    return active_stopwords, stop_word_config


@app.cell(hide_code=True)
def _(active_stopwords, frequent_terms, mo, modelled_corpus, pd):
    _count = mo.md(f"**{len(active_stopwords)} stop words** are active.")
    if modelled_corpus is None:
        _view = _count
    else:
        _terms = pd.DataFrame(
            frequent_terms(modelled_corpus.documents, active_stopwords, top_n=20),
            columns=["term", "count"],
        )
        _words = pd.DataFrame({"stop word": sorted(active_stopwords)})
        _view = mo.vstack(
            [
                _count,
                mo.accordion(
                    {
                        "Frequent terms after cleaning": mo.ui.table(_terms, selection=None),
                        "Active stop words": mo.ui.table(_words, selection=None),
                    }
                ),
            ]
        )
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    get_reset, set_reset = mo.state(0)
    # Holds (config, file name, failure) for a loaded config.json, or None. The reset button
    # clears it, so a reset always restores the recommended defaults.
    get_loaded, set_loaded = mo.state(None)
    # Maps a model setting to the value that the reader chose. A change of the data source
    # rebuilds the controls below, and these values survive it. A reset and a loaded config.json
    # clear it.
    get_touched, set_touched = mo.state({})
    return get_loaded, get_reset, get_touched, set_loaded, set_reset, set_touched


@app.cell(hide_code=True)
def _(get_loaded, get_reset, get_touched, mo, set_loaded, set_reset, set_touched, source):
    get_reset()
    _loaded = get_loaded()
    _model = _loaded[0].model if _loaded is not None and _loaded[0] is not None else None
    _touched = get_touched()

    def _start(field, default):
        # The reader's own choice wins over a loaded config.json, which wins over the default.
        if field in _touched:
            return _touched[field]
        return getattr(_model, field) if _model is not None else default

    def _keep(field):
        # A setter called from an element of this cell does not rerun this cell, so a drag on a
        # slider never rebuilds it.
        def _store(value):
            set_touched(lambda current: {**current, field: value})

        return _store

    _model_labels = {"nmf": "NMF (recommended)", "lda": "LDA"}
    _ngram_labels = {"1": "Single words", "1-2": "Words and pairs", "2": "Pairs only"}
    model_type = mo.ui.dropdown(
        options={"NMF (recommended)": "nmf", "LDA": "lda"},
        value=_model_labels[_start("model_type", "nmf")],
        label="Model",
        on_change=_keep("model_type"),
    )
    # The demo corpus carries six newspaper sections. Six topics therefore give a first result
    # that a newcomer can check against the category column. Ten topics split the finance
    # articles into several topics of bare numbers. The source decides the count only until the
    # reader moves the slider.
    _start_topics = 6 if source.value == "Demo data" else 10
    n_topics = mo.ui.slider(
        2,
        30,
        value=_start("n_topics", _start_topics),
        step=1,
        label="Number of topics",
        show_value=True,
        on_change=_keep("n_topics"),
    )
    max_features = mo.ui.number(
        100,
        20000,
        value=_start("max_features", 5000),
        step=100,
        label="Max vocabulary",
        on_change=_keep("max_features"),
    )
    min_df = mo.ui.number(
        1,
        100,
        value=_start("min_df", 2),
        step=1,
        label="Minimum document frequency",
        on_change=_keep("min_df"),
    )
    max_df = mo.ui.slider(
        0.5,
        1.0,
        value=_start("max_df", 0.95),
        step=0.01,
        label="Maximum document frequency",
        show_value=True,
        on_change=_keep("max_df"),
    )
    ngrams = mo.ui.dropdown(
        options={"Single words": "1", "Words and pairs": "1-2", "Pairs only": "2"},
        value=_ngram_labels[_start("ngrams", "1")],
        label="N-grams",
        on_change=_keep("ngrams"),
    )
    # The seed is what makes a run repeatable. It already reached both models; it was only
    # invisible. A reader who has the corpus and the seed gets the same topics.
    random_seed = mo.ui.number(
        0,
        2**31 - 1,
        value=_start("random_seed", 42),
        step=1,
        label="Random seed",
        on_change=_keep("random_seed"),
    )

    def _reset(_value):
        set_loaded(None)
        set_touched({})
        set_reset(lambda current: current + 1)

    reset_button = mo.ui.button(label="Reset recommended defaults", on_change=_reset)
    return max_df, max_features, min_df, model_type, n_topics, ngrams, random_seed, reset_button


@app.cell(hide_code=True)
def _(
    max_df,
    max_features,
    min_df,
    mo,
    model_type,
    n_topics,
    ngrams,
    random_seed,
    reset_button,
    source,
):
    _model_help = mo.accordion(
        {
            "When to use NMF or LDA": mo.md(
                """
            **NMF** is the place to start for most collections. It is fast, it works well with
            short and medium documents, and it usually gives readable keyword topics. Its topic
            scores are relative weights, not probabilities.

            **LDA** suits medium and long documents, and it gives a probabilistic topic mixture.
            It is slower and more sensitive to the parameters. Use NMF if you are unsure.
            """
            )
        }
    )
    _note = (
        mo.md(
            "*The newspaper sorted the demo articles into six sections, so the app starts at six"
            " topics. Move the slider to see the themes split or merge.*"
        )
        if source.value == "Demo data"
        else mo.md("")
    )
    _advanced = mo.accordion(
        {
            "Advanced settings": mo.vstack(
                [
                    mo.hstack(
                        [max_features, min_df, max_df, ngrams, random_seed],
                        justify="start",
                        gap=2,
                        wrap=True,
                    ),
                    reset_button,
                    mo.md(
                        """
                    **Number of topics.** How many themes the model looks for. Start around 8 to
                    12 for a corpus that you do not know. Raise it if the topics are too broad.
                    Lower it if many topics look alike or tiny.

                    **Max vocabulary.** How many distinct terms the model considers. 5,000 is a
                    good browser-friendly default. Raise it for a large, varied corpus. Lower it
                    for speed.

                    **Minimum document frequency.** Ignore a word that occurs in fewer than this
                    many documents. Raise it to drop typos, names, and very rare terms.

                    **Maximum document frequency.** Ignore a word that occurs in almost every
                    document. `0.95` ignores a word that appears in more than 95% of documents.

                    **N-grams.** Single words, or also two-word phrases such as *climate change*.
                    Phrases can improve a label, but they make the vocabulary larger and slower.

                    **Random seed.** The number that fixes the starting point of the model. The
                    same seed on the same corpus and the same settings gives the same topics.
                    Change it to see how stable your topics are.
                    """
                    ),
                ]
            )
        }
    )
    mo.vstack(
        [
            mo.hstack([model_type, n_topics], justify="start", gap=2),
            _note,
            _model_help,
            _advanced,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    AppConfig,
    ModelConfig,
    PreprocessConfig,
    analyse_as,
    language,
    max_df,
    max_features,
    min_df,
    model_type,
    n_topics,
    ngrams,
    random_seed,
    split_mode,
    stop_word_config,
    table,
    text_documents,
):
    # The split settings apply to one text only. Any other input records no split.
    _one_text = table is None and text_documents is not None and len(text_documents) == 1
    pending_config = AppConfig(
        analyse_as=analyse_as.value if _one_text else "corpus",
        split_mode=split_mode.value if _one_text else None,
        language=language.value,
        preprocess=PreprocessConfig(),
        stop_words=stop_word_config,
        model=ModelConfig(
            model_type=model_type.value,
            n_topics=n_topics.value,
            max_features=int(max_features.value),
            min_df=int(min_df.value),
            max_df=float(max_df.value),
            ngrams=ngrams.value,
            random_seed=int(random_seed.value),
        ),
    )
    return (pending_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 3 — Run and explore""")
    return


@app.cell(hide_code=True)
def _(corpus, mo, pending_config):
    run_button = mo.ui.run_button(label="Run model", kind="success", disabled=corpus is None)
    # The sidebar stays in view while the page scrolls, so the summary and the run button sit
    # next to the settings and the results alike.
    _summary = mo.md(f"**{pending_config.summary(len(corpus) if corpus else 0)}**")
    mo.sidebar([mo.md("### Run"), _summary, run_button], width="20rem")
    return (run_button,)


@app.cell(hide_code=True)
def _(mo):
    get_result, set_result = mo.state(None)
    get_failure, set_failure = mo.state(None)
    get_overrides, set_overrides = mo.state({})
    # One value holds the selected topic for every view. The dropdown, the topic map, the bars,
    # and the card buttons all write it. Every view reads it.
    get_selected, set_selected = mo.state(None)
    return (
        get_failure,
        get_overrides,
        get_result,
        get_selected,
        set_failure,
        set_overrides,
        set_result,
        set_selected,
    )


@app.cell(hide_code=True)
def _(
    FriendlyMessage,
    TopicError,
    get_run_request,
    modeling,
    modelled_corpus,
    pending_config,
    run_button,
    set_failure,
    set_overrides,
    set_result,
    set_run_request,
):
    # This cell holds the one expensive step. It fits only after the run button or a run request.
    # The cell clears the run request first, so one request starts one fit.
    # marimo resets `run_button.value` to False after the dependent cells run, so a change to any
    # setting re-runs this cell without fitting and the previous result survives. A failed fit
    # stores a message and leaves the previous result in place.
    _requested = get_run_request()
    if _requested:
        set_run_request(False)
    if (run_button.value or _requested) and modelled_corpus is not None:
        try:
            set_result(modeling.fit_topic_model(modelled_corpus, pending_config))
            set_overrides({})
            set_failure(None)
        except TopicError as error:
            set_failure(error.friendly)
        except (MemoryError, ValueError, ArithmeticError):
            set_failure(
                FriendlyMessage(
                    "The browser could not finish the model fit.",
                    "Use fewer documents, lower the maximum vocabulary, or ask for fewer topics.",
                )
            )
    return


@app.cell(hide_code=True)
def _(get_overrides, get_result, result_mod):
    display_result = get_result()
    if display_result is not None:
        for _index, _name in get_overrides().items():
            display_result = result_mod.rename_topic(display_result, _index, _name)
    return (display_result,)


@app.cell(hide_code=True)
def _(config_changed, display_result, get_failure, mo, pending_config, set_run_request):
    # The rerun button needs a global name: marimo holds a UI element by a weak reference.
    # It sets the same run request as Run demo, and the fit cell clears that request.
    rerun_button = mo.ui.button(
        label="Rerun",
        kind="warn",
        on_change=lambda _value: set_run_request(True),
    )
    _notes = []
    _failure = get_failure()
    if _failure is not None:
        _notes.append(
            mo.callout(mo.md(f"**{_failure.detail}** {_failure.recovery}"), kind="danger")
        )
    if config_changed(None if display_result is None else display_result.config, pending_config):
        _notes.append(
            mo.callout(
                mo.hstack(
                    [mo.md("**Configuration changed — rerun to update results.**"), rerun_button],
                    justify="space-between",
                    align="center",
                ),
                kind="warn",
            )
        )
    mo.vstack(_notes) if _notes else mo.md("")
    return (rerun_button,)


@app.cell(hide_code=True)
def _(display_result, get_selected, resolve_topic):
    # The stored selection can be stale after a new fit, so it is clamped to a valid topic.
    selected_index = (
        resolve_topic(get_selected(), len(display_result.topic_names))
        if display_result is not None
        else None
    )
    return (selected_index,)


@app.cell(hide_code=True)
def _(display_result, get_selected, mo, selected_index, set_selected):
    def _choose(_value):
        # Write only a changed value, so a redraw never starts a loop.
        if _value is not None and _value != get_selected():
            set_selected(_value)

    _options = (
        {name: number for number, name in enumerate(display_result.topic_names)}
        if display_result is not None
        else {}
    )
    _current = (
        display_result.topic_names[selected_index]
        if display_result is not None and selected_index is not None
        else None
    )
    topic_select = mo.ui.dropdown(
        options=_options, value=_current, label="Selected topic", on_change=_choose
    )
    return (topic_select,)


@app.cell(hide_code=True)
def _(display_result, mo, plots, selected_index):
    # marimo keeps only weak references to UI elements, so each chart has a global name.
    # The charts are rebuilt with the selected topic, so every view shows the same highlight.
    # A click on a bubble or a bar is read in the next cell.
    map_chart = bars_chart = None
    if display_result is not None:
        map_chart = mo.ui.altair_chart(plots.topic_map(display_result, selected_index))
        bars_chart = mo.ui.altair_chart(plots.prevalence_bars(display_result, selected_index))
    return bars_chart, map_chart


@app.cell(hide_code=True)
def _(display_result, get_selected, mo, set_selected):
    def _pick(_index):
        def _click(_count):
            if _index != get_selected():
                set_selected(_index)
            return (_count or 0) + 1

        return _click

    # One button per card, held in a global name. The buttons stay when the selection changes.
    # Each label carries the topic number, so a screen reader can tell the buttons apart. The
    # number stays unique after a rename.
    card_buttons = (
        mo.ui.array(
            [
                mo.ui.button(label=f"Select topic {_number + 1}", on_click=_pick(_number), value=0)
                for _number in range(len(display_result.topic_names))
            ]
        )
        if display_result is not None
        else None
    )
    return (card_buttons,)


@app.cell(hide_code=True)
def _(mo, selected_index, set_overrides):
    # The rename acts on the shared selection. A new selection rebuilds the input, so it starts
    # empty. An empty name restores the automatic label. The overrides apply in order, and a
    # taken name gets a number. The latest rename therefore moves to the end, so the topic that
    # the reader just renamed gets the number, not an earlier one.
    def _apply(_value):
        if selected_index is not None:
            set_overrides(
                lambda current: {
                    **{index: name for index, name in current.items() if index != selected_index},
                    selected_index: rename_input.value,
                }
            )

    rename_input = mo.ui.text(label="Rename the selected topic", placeholder="Economy")
    rename_button = mo.ui.button(label="Apply name", on_change=_apply)
    return rename_button, rename_input


@app.cell(hide_code=True)
def _(
    bars_chart,
    display_result,
    get_selected,
    map_chart,
    selected_topic_from_chart,
    set_selected,
):
    # A chart click writes the shared selection. Both charts are rebuilt after each change, so
    # their value is empty until the next click. The write happens only for a changed value.
    if display_result is not None:
        for _chart in (map_chart, bars_chart):
            _picked = selected_topic_from_chart(
                _chart.value, n_topics=len(display_result.topic_names)
            )
            if _picked is not None and _picked != get_selected():
                set_selected(_picked)
    return


@app.cell(hide_code=True)
def _(display_result, mo):
    _names = display_result.topic_names if display_result is not None else []
    topic_filter = mo.ui.multiselect(options=_names, value=[], label="Show only these topics")
    score_filter = mo.ui.slider(
        0.0, 1.0, value=0.0, step=0.05, label="Minimum dominant-topic score", show_value=True
    )
    document_search = mo.ui.text(
        label="Search the text", placeholder="word or phrase, then press Enter"
    )
    return document_search, score_filter, topic_filter


@app.cell(hide_code=True)
def _(
    bars_chart,
    card_buttons,
    display_result,
    document_search,
    map_chart,
    metrics,
    mo,
    pd,
    plots,
    rename_button,
    rename_input,
    score_filter,
    selected_index,
    topic_filter,
    topic_select,
):
    if display_result is None:
        _view = mo.md(
            "*No result yet. Go to **Step 3** and select **Run model** to build the topics.*"
        )
    else:
        _cards = []
        for _row in plots.topic_cards(display_result).itertuples():
            _chosen = _row.topic_id == selected_index
            _card = mo.vstack(
                [
                    # A custom name can hold Markdown characters, so the card shows it verbatim.
                    mo.Html(
                        plots.topic_card_html(
                            _row.topic, _row.top_terms, _row.prevalence, _row.documents, _chosen
                        )
                    ),
                    card_buttons[_row.topic_id],
                ],
                gap=0.5,
            ).style(
                {
                    "border": f"{3 if _chosen else 1}px solid currentColor",
                    "opacity": "1" if _chosen else "0.7",
                    "padding": "0.75rem",
                    "border-radius": "6px",
                    "min-width": "14rem",
                    "max-width": "18rem",
                }
            )
            _cards.append(_card)
        _overview = mo.vstack(
            [
                mo.md("### Topics at a glance"),
                mo.md(
                    "*Select a topic with its card button, a bubble on the map, a bar, or the"
                    " dropdown. Every view then highlights the same topic.*"
                ),
                mo.hstack(_cards, justify="start", gap=1, wrap=True),
                map_chart,
                bars_chart,
                mo.ui.altair_chart(plots.similarity_heatmap(display_result, selected_index)),
            ]
        )

        _index = selected_index
        # The fitted result, not the live control, decides the mode. A changed control only
        # raises the "Configuration changed" banner until the next run.
        _long = display_result.config.get("analyse_as") == "long_document"
        _unit = "segments" if _long else "documents"
        if _long:
            _positions = plots.position_frame(display_result)
            # Empty paragraphs and sampling leave gaps, so the row count is not the last position.
            _last = int(display_result.metadata["segment_number"].max())
            _examples = [
                mo.ui.altair_chart(plots.topic_position_area(_positions, _index)),
                mo.md("### Representative passages"),
                mo.md(
                    "*The segment number is the position of the passage in the source. Segment 1"
                    f" opens the text. Segment {_last} is the last one"
                    " that the model used.*"
                ),
                mo.ui.table(plots.representative_passages(display_result, _index), selection=None),
            ]
            _position_view = [
                mo.ui.altair_chart(plots.position_heatmap(_positions, display_result.topic_names)),
                mo.md(
                    "*Each column is one position in the text. A darker cell means a larger"
                    " topic share. A long text averages neighbouring segments into one column.*"
                ),
            ]
        else:
            _examples = [
                mo.md("### Representative documents"),
                mo.ui.table(plots.representative_documents(display_result, _index), selection=None),
            ]
            _position_view = []
        _topics = mo.vstack(
            [
                mo.hstack(
                    [topic_select, rename_input, rename_button], justify="start", align="end"
                ),
                mo.md(
                    f"**Prevalence:** {display_result.topic_prevalence[_index]:.1%} of the"
                    f" {'text' if _long else 'corpus'}"
                    f" · **{display_result.n_documents}** {_unit} modelled"
                ),
                mo.ui.altair_chart(plots.top_term_bars(display_result, _index)),
                mo.image(plots.word_cloud_png(display_result, _index), width=700),
                *_examples,
                mo.accordion(
                    {
                        "How to read a topic": mo.md(
                            f"""
                        Read the top terms together with several high-scoring
                        {"passages" if _long else "documents"}. The keywords are clues, not a full
                        definition. Rename the topic once its meaning is clear to you.
                        """
                        )
                    }
                ),
            ]
        )

        _documents = plots.document_frame(display_result)
        if topic_filter.value:
            _documents = _documents[_documents["topic"].isin(topic_filter.value)]
        if score_filter.value > 0:
            _documents = _documents[_documents["score"] >= score_filter.value]
        if document_search.value.strip():
            _documents = _documents[
                _documents["snippet"].str.contains(
                    document_search.value.strip(), case=False, regex=False
                )
            ]
        _documents = _documents.reset_index(drop=True)
        if _documents.empty:
            _document_view = mo.callout(
                mo.md(f"**No {_unit[:-1]} matches these filters.** Widen them to see results."),
                kind="warn",
            )
        else:
            _document_view = mo.vstack(
                [
                    mo.ui.altair_chart(plots.document_scatter(_documents)),
                    mo.ui.table(_documents.drop(columns=["x", "y"]), selection=None),
                ]
            )
        _explore = mo.vstack(
            [
                *_position_view,
                mo.hstack(
                    [topic_filter, score_filter, document_search],
                    justify="start",
                    gap=2,
                    wrap=True,
                ),
                mo.md(f"**{len(_documents)}** of **{display_result.n_documents}** {_unit} shown."),
                _document_view,
            ]
        )
        _tabs = {"Overview": _overview, "Topics": _topics, "Documents": _explore}

        _columns = list(display_result.metadata.columns)
        _panels = []
        if "group" in _columns:
            _panels.append(
                mo.ui.altair_chart(
                    plots.group_stacked_bars(plots.group_share_frame(display_result, "group"))
                )
            )
        if "date" in _columns:
            _parsed, _unparsed = plots.parse_dates(display_result.metadata["date"])
            _readable = len(_parsed) - _unparsed
            _panels.append(mo.md(f"**{_readable}** of **{len(_parsed)}** dates were readable."))
            if _unparsed:
                _panels.append(
                    mo.callout(
                        mo.md(
                            f"**The app could not read {_unparsed} of {len(_parsed)} dates.**"
                            " Use the ISO format YYYY-MM-DD, or choose another column."
                        ),
                        kind="warn",
                    )
                )
            if _readable:
                _bin = plots.choose_date_bin(_parsed)
                _panels.append(mo.md(f"Binned by **{_bin}**."))
                _panels.append(
                    mo.ui.altair_chart(
                        plots.time_line_chart(plots.time_share_frame(display_result, _parsed, _bin))
                    )
                )
        if _panels:
            _tabs["Metadata"] = mo.vstack(_panels)

        _report = metrics.diagnostics(display_result)
        _numbers = pd.DataFrame(
            {
                "Documents used": [_report["documents_used"]],
                "Vocabulary": [_report["vocabulary_size"]],
                "Topics": [_report["topic_count"]],
                "Topic diversity": [f"{_report['topic_diversity']:.2f}"],
                "Mean pairwise similarity": [f"{_report['mean_pairwise_similarity']:.2f}"],
                "Weak dominant scores": [f"{_report['weak_dominant_share']:.0%}"],
            }
        )
        _quality = (
            f"Reconstruction error: {_report['reconstruction_error']:.3f}"
            if _report["reconstruction_error"] is not None
            else f"Perplexity: {_report['perplexity']:.1f}"
        )
        _tabs["Diagnostics"] = mo.vstack(
            [
                mo.ui.table(_numbers, selection=None),
                mo.md(f"**{_quality}**"),
                *[
                    mo.callout(mo.md(f"**{_note.detail}** {_note.recovery}"), kind="warn")
                    for _note in _report["notices"]
                ],
                mo.ui.altair_chart(
                    plots.score_histogram(
                        pd.DataFrame({"score": display_result.dominant_topic_score})
                    )
                ),
                mo.accordion(
                    {
                        "How to read the diagnostics": mo.md(
                            """
                        These numbers describe the run. They are not a quality score. A model is
                        good when its topics help you answer your question.
                        """
                        )
                    }
                ),
            ]
        )
        _position_note = (
            """
        This run analyses **one ordered text**. Each segment is one piece of the text, usually a
        paragraph, or a page for a PDF. Its number is its position. **Topics** shows where the
        selected topic occurs and lists its representative passages. **Documents** shows the topic
        share through the whole text, then lists the segments in their order.
        """
            if _long
            else ""
        )
        _orientation = mo.md(
            f"""
        **Overview** shows every topic at once. **Topics** opens one topic in detail.
        **Documents** lists the {_unit} of a topic. **Metadata** charts the topics against your
        own columns. **Diagnostics** describes the run.
        {_position_note}
        Start in **Topics**. Read the terms of a topic. Give the topic a name. Then go to Step 4
        and export your results.
        """
        )
        _glossary = mo.accordion(
            {
                "Glossary": mo.md(
                    """
                **Topic.** A group of words that occur together in the corpus.

                **Term.** One word or phrase in a topic. The model gives each term a weight.

                **Prevalence.** The share of the corpus that one topic covers.

                **Dominant score.** The score of the strongest topic of one document. A low score
                means that the document fits no topic well.

                **Segment.** One piece of an ordered text, such as a paragraph. Its number is
                its position in the text.

                **Topic diversity.** The share of top terms that occur in one topic only. A low
                value means that the topics repeat each other.

                **Document frequency.** The number of documents that hold a term. The app uses it
                to drop a term that is too rare or too common.

                **N-gram.** A run of words that the model treats as one term. *Chronique
                militaire* is a 2-gram.

                **TF-IDF.** A weight for a term. It rises when the term occurs often in one
                document. It falls when the term occurs in many documents.
                """
                )
            }
        )
        _view = mo.vstack([_orientation, mo.ui.tabs(_tabs), _glossary])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 4 — Export""")
    return


@app.cell(hide_code=True)
def _(mo):
    include_text = mo.ui.checkbox(value=False, label="Include the document text in the CSV")
    include_text
    return (include_text,)


@app.cell(hide_code=True)
def _(AppConfig, display_result, exports, include_text, mo):
    if display_result is None:
        _view = mo.md(
            "*No result yet. Go to **Step 3** and select **Run model** to unlock the downloads.*"
        )
    else:
        # Export the settings of the fitted result, not the live controls. A changed control
        # would otherwise describe a run that did not make these files.
        _config = AppConfig.model_validate(display_result.config)
        _files = {
            "documents_topics.csv": exports.documents_topics_frame(
                display_result, include_text.value
            ),
            "topics.csv": exports.topics_frame(display_result),
            "topic_terms.csv": exports.topic_terms_frame(display_result),
            "topic_similarity.csv": exports.topic_similarity_frame(display_result),
        }
        _buttons = [
            mo.download(
                data=exports.to_csv_bytes(frame),
                filename=name,
                label=name,
                mimetype="text/csv",
            )
            for name, frame in _files.items()
        ]
        _buttons.append(
            mo.download(
                data=exports.config_json(_config, display_result.topic_names),
                filename="config.json",
                label="config.json",
                mimetype="application/json",
            )
        )
        _buttons.append(
            mo.download(
                data=exports.project_zip(display_result, _config, include_text.value),
                filename="project.zip",
                label="project.zip",
                mimetype="application/zip",
            )
        )
        _view = mo.hstack(_buttons, justify="start", gap=1, wrap=True)
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ---

    ### About

    **The tool.** [Moritz Mähr](https://github.com/maehr) wrote Simple Topic Modeling. It is free
    software under the
    [AGPL-3.0](https://github.com/maehr/simple-topic-modeling/blob/main/LICENSE).

    **The demo corpus.** The corpus holds 295 articles from the *Journal de Genève* and the
    *Gazette de Lausanne* of 1914. The Digital Humanities Laboratory of the EPFL digitised the
    historical archive of *Le Temps*. It published the year 1914 under CC BY 4.0, for the 2015
    Swiss Open Cultural Data Hackathon. The articles are anonymous newspaper text from 1914, so
    they left copyright in 1985. The
    [project page](https://hack.glam.opendata.ch/project/234) holds the archive, and
    [`NOTICE`](https://github.com/maehr/simple-topic-modeling/blob/main/NOTICE) holds the full
    statement.

    **Take part.** [Report a problem or ask for a
    feature](https://github.com/maehr/simple-topic-modeling/issues). Read the [contribution
    guidelines](https://github.com/maehr/simple-topic-modeling/blob/main/CONTRIBUTING.md). Read
    the [source](https://github.com/maehr/simple-topic-modeling).
    """
    )
    return


if __name__ == "__main__":
    app.run()
