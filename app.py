# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair",
#     "numpy",
#     "pandas",
#     "pillow",
#     "pydantic",
#     "scikit-learn",
#     "scipy",
#     "wordcloud",
#     "browser-topics @ public/wheels/browser_topics-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Browser Topic Explorer")


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
            mo.notebook_location() / "public" / "wheels" / "browser_topics-0.1.0-py3-none-any.whl"
        )
        await micropip.install(str(_wheel))

    import altair as alt
    import pandas as pd

    from browser_topics import exports, io, modeling, plots
    from browser_topics import result as result_mod
    from browser_topics.config import (
        LANGUAGE_LABELS,
        AppConfig,
        ModelConfig,
        PreprocessConfig,
        StopWordConfig,
    )
    from browser_topics.errors import FriendlyMessage, TopicError
    from browser_topics.preprocess import frequent_terms
    from browser_topics.stopwords import effective_stopwords

    return (
        AppConfig,
        FriendlyMessage,
        LANGUAGE_LABELS,
        ModelConfig,
        PreprocessConfig,
        StopWordConfig,
        TopicError,
        alt,
        effective_stopwords,
        exports,
        frequent_terms,
        io,
        mo,
        modeling,
        pd,
        plots,
        result_mod,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Browser Topic Explorer

    Find the themes in a collection of documents.
    """
    )
    return


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
    mo.md("""## Step 1 — Add data""")
    return


@app.cell(hide_code=True)
def _(mo):
    source = mo.ui.radio(
        options=["Demo data", "Upload files", "Paste text"],
        value="Demo data",
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

            PDF, DOCX, images, and ZIP files are out of scope.
            """
            )
        }
    )
    mo.vstack([_widget, _help]) if _widget is not None else _help
    return file_input, paste_input


@app.cell(hide_code=True)
def _(TopicError, file_input, io, paste_input, source):
    table = None
    text_documents = None
    text_names = None
    load_error = None

    try:
        if source.value == "Demo data":
            table = io.demo_table()
        elif source.value == "Upload files":
            _uploads = [io.UploadedFile(item.name, item.contents) for item in file_input.value]
            _tables = [item for item in _uploads if io.detect_kind(item.name) != "text"]
            _plain = [item for item in _uploads if io.detect_kind(item.name) == "text"]
            if _tables:
                table = io.read_table(_tables[0])
            elif _plain:
                text_documents = [
                    io.strip_markup(io.decode_text(item), item.name) for item in _plain
                ]
                text_names = [item.name for item in _plain]
        elif paste_input.value.strip():
            text_documents = [paste_input.value]
            text_names = ["pasted text"]
    except TopicError as error:
        load_error = error.friendly

    return load_error, table, text_documents, text_names


@app.cell(hide_code=True)
def _(mo, table, text_documents, text_names):
    _columns = [str(name) for name in table.columns] if table is not None else []
    _guess = next(
        (name for name in _columns if name.lower() in {"text", "body", "content", "abstract"}),
        _columns[0] if _columns else None,
    )
    text_column = mo.ui.dropdown(options=_columns, value=_guess, label="Text column")
    id_column = mo.ui.dropdown(
        options=["(row number)", *_columns], value="(row number)", label="Document ID"
    )
    date_column = mo.ui.dropdown(options=["(none)", *_columns], value="(none)", label="Date column")
    group_column = mo.ui.dropdown(
        options=["(none)", *_columns], value="(none)", label="Group column"
    )
    split_mode = mo.ui.dropdown(
        options={
            "Whole file is one document": "whole",
            "Split on blank lines": "blank_lines",
            "Split by line": "lines",
        },
        value="Split on blank lines",
        label="How should the app split this text?",
    )

    if table is not None:
        _controls = mo.hstack(
            [text_column, id_column, date_column, group_column], justify="start", gap=1, wrap=True
        )
    elif text_documents is not None and len(text_documents) == 1:
        _controls = split_mode
    elif text_names:
        _controls = mo.md(f"**{len(text_names)} files.** One file is one document.")
    else:
        _controls = mo.md("*Add data to continue.*")
    _controls
    return date_column, group_column, id_column, split_mode, text_column


@app.cell(hide_code=True)
def _(
    TopicError,
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
                _pieces = io.split_text(text_documents[0], split_mode.value)
                _identifiers = [f"{text_names[0]}#{number + 1}" for number in range(len(_pieces))]
                corpus, stats = io.build_corpus(_pieces, _identifiers)
            else:
                corpus, stats = io.build_corpus(text_documents, list(text_names))
    except TopicError as error:
        corpus_error = error.friendly

    return corpus, corpus_error, stats


@app.cell(hide_code=True)
def _(corpus, corpus_error, load_error, mo, pd, stats):
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
        _view = mo.vstack(
            [
                mo.ui.table(_numbers, selection=None),
                mo.md("**Preview**"),
                mo.ui.table(_preview, selection=None),
            ]
        )
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 2 — Configure""")
    return


@app.cell(hide_code=True)
def _(LANGUAGE_LABELS, mo):
    language = mo.ui.dropdown(
        options={label: code for code, label in LANGUAGE_LABELS.items()},
        value="English",
        label="Language",
    )
    use_base = mo.ui.checkbox(value=True, label="Use the base stop-word list")
    added_words = mo.ui.text_area(
        label="Add stop words", placeholder="survey, january, ltd", rows=3
    )
    keep_words = mo.ui.text_area(label="Always keep these words", placeholder="growth", rows=3)
    _help = mo.accordion(
        {
            "About the language setting": mo.md(
                """
            Choose the main language of your documents. This mainly changes the common words that
            the app ignores, such as *the*, *und*, *le*, *di*, or *el*.

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
def _(active_stopwords, corpus, frequent_terms, mo, pd):
    _count = mo.md(f"**{len(active_stopwords)} stop words** are active.")
    if corpus is None:
        _view = _count
    else:
        _terms = pd.DataFrame(
            frequent_terms(corpus.documents, active_stopwords, top_n=20),
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
    return get_reset, set_reset


@app.cell(hide_code=True)
def _(get_reset, mo, set_reset):
    get_reset()
    model_type = mo.ui.dropdown(
        options={"NMF (recommended)": "nmf", "LDA": "lda"},
        value="NMF (recommended)",
        label="Model",
    )
    n_topics = mo.ui.slider(2, 30, value=10, step=1, label="Number of topics", show_value=True)
    max_features = mo.ui.number(100, 20000, value=5000, step=100, label="Max vocabulary")
    min_df = mo.ui.number(1, 100, value=2, step=1, label="Minimum document frequency")
    max_df = mo.ui.slider(
        0.5, 1.0, value=0.95, step=0.01, label="Maximum document frequency", show_value=True
    )
    ngrams = mo.ui.dropdown(
        options={"Single words": "1", "Words and pairs": "1-2", "Pairs only": "2"},
        value="Single words",
        label="N-grams",
    )
    reset_button = mo.ui.button(
        label="Reset recommended defaults",
        on_change=lambda _value: set_reset(lambda current: current + 1),
    )
    return max_df, max_features, min_df, model_type, n_topics, ngrams, reset_button


@app.cell(hide_code=True)
def _(max_df, max_features, min_df, mo, model_type, n_topics, ngrams, reset_button):
    _help = mo.accordion(
        {
            "When to use NMF or LDA": mo.md(
                """
            **NMF** is the place to start for most collections. It is fast, it works well with
            short and medium documents, and it usually gives readable keyword topics. Its topic
            scores are relative weights, not probabilities.

            **LDA** suits medium and long documents, and it gives a probabilistic topic mixture.
            It is slower and more sensitive to the parameters. Use NMF if you are unsure.
            """
            ),
            "What the parameters do": mo.md(
                """
            **Number of topics.** How many themes the model looks for. Start around 8 to 12.
            Raise it if the topics are too broad. Lower it if many topics look alike or tiny.

            **Max vocabulary.** How many distinct terms the model considers. 5,000 is a good
            browser-friendly default. Raise it for a large, varied corpus. Lower it for speed.

            **Minimum document frequency.** Ignore a word that occurs in fewer than this many
            documents. Raise it to drop typos, names, and very rare terms.

            **Maximum document frequency.** Ignore a word that occurs in almost every document.
            `0.95` ignores a word that appears in more than 95% of documents.

            **N-grams.** Single words, or also two-word phrases such as *climate change*. Phrases
            can improve a label, but they make the vocabulary larger and slower.

            **Iterations.** Usually leave this alone. Raise it only if the app reports that the
            model did not converge.
            """
            ),
        }
    )
    mo.vstack(
        [
            mo.hstack([model_type, n_topics], justify="start", gap=2),
            mo.hstack([max_features, min_df, max_df, ngrams], justify="start", gap=2, wrap=True),
            reset_button,
            _help,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    AppConfig,
    ModelConfig,
    PreprocessConfig,
    language,
    max_df,
    max_features,
    min_df,
    model_type,
    n_topics,
    ngrams,
    stop_word_config,
):
    pending_config = AppConfig(
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
        ),
    )
    return (pending_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 3 — Run the model""")
    return


@app.cell(hide_code=True)
def _(corpus, mo, pending_config):
    run_button = mo.ui.run_button(label="Run model", kind="success", disabled=corpus is None)
    mo.vstack([mo.md(f"**{pending_config.summary(len(corpus) if corpus else 0)}**"), run_button])
    return (run_button,)


@app.cell(hide_code=True)
def _(mo):
    get_result, set_result = mo.state(None)
    get_failure, set_failure = mo.state(None)
    get_overrides, set_overrides = mo.state({})
    return get_failure, get_overrides, get_result, set_failure, set_overrides, set_result


@app.cell(hide_code=True)
def _(
    FriendlyMessage,
    TopicError,
    corpus,
    modeling,
    pending_config,
    run_button,
    set_failure,
    set_overrides,
    set_result,
):
    # This cell holds the one expensive step. It fits only when the run button was clicked.
    # marimo resets `run_button.value` to False after the dependent cells run, so a change to any
    # setting re-runs this cell without fitting and the previous result survives. A failed fit
    # stores a message and leaves the previous result in place.
    if run_button.value and corpus is not None:
        try:
            set_result(modeling.fit_topic_model(corpus, pending_config))
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
def _(display_result, get_failure, mo, pending_config):
    _notes = []
    _failure = get_failure()
    if _failure is not None:
        _notes.append(
            mo.callout(mo.md(f"**{_failure.detail}** {_failure.recovery}"), kind="danger")
        )
    if display_result is not None and display_result.config != pending_config.model_dump():
        _notes.append(
            mo.callout(mo.md("**Configuration changed — rerun to update results.**"), kind="warn")
        )
    mo.vstack(_notes) if _notes else mo.md("")
    return


@app.cell(hide_code=True)
def _(display_result, mo):
    _options = (
        {name: number for number, name in enumerate(display_result.topic_names)}
        if display_result is not None
        else {}
    )
    topic_select = mo.ui.dropdown(
        options=_options, value=next(iter(_options), None), label="Selected topic"
    )
    return (topic_select,)


@app.cell(hide_code=True)
def _(display_result, mo, topic_select):
    rename_input = mo.ui.text(label="Rename the selected topic", placeholder="Economy")
    _view = mo.md("") if display_result is None or topic_select.value is None else rename_input
    _view
    return (rename_input,)


@app.cell(hide_code=True)
def _(display_result, mo, rename_input, set_overrides, topic_select):
    def _apply(_value):
        if topic_select.value is not None:
            set_overrides(lambda current: {**current, topic_select.value: rename_input.value})

    _button = mo.ui.button(label="Apply name", on_change=_apply)
    _button if display_result is not None else mo.md("")
    return


@app.cell(hide_code=True)
def _(alt, display_result, mo, plots, topic_select):
    if display_result is None:
        _view = mo.md("*Run the model to explore the topics.*")
    else:
        _overview = mo.vstack(
            [
                mo.md("### Topics at a glance"),
                mo.ui.table(plots.topic_cards(display_result), selection=None, page_size=30),
            ]
        )
        _index = topic_select.value if topic_select.value is not None else 0
        _terms = plots.top_term_frame(display_result, _index)
        _bars = (
            alt.Chart(_terms, title=f"Top terms · {display_result.topic_names[_index]}")
            .mark_bar()
            .encode(
                x=alt.X("weight:Q", title="Share of the topic"),
                y=alt.Y("term:N", sort="-x", title=None),
                tooltip=["term:N", alt.Tooltip("weight:Q", format=".3f")],
            )
            .properties(height=300)
        )
        _topics = mo.vstack(
            [
                topic_select,
                mo.md(
                    f"**Prevalence:** {display_result.topic_prevalence[_index]:.1%} of the corpus"
                    f" · **{display_result.n_documents}** documents modelled"
                ),
                mo.ui.altair_chart(_bars),
                mo.md("### Representative documents"),
                mo.ui.table(plots.representative_documents(display_result, _index), selection=None),
                mo.accordion(
                    {
                        "How to read a topic": mo.md(
                            """
                        Read the top terms together with several high-scoring documents. The
                        keywords are clues, not a full definition. Rename the topic once its
                        meaning is clear to you.
                        """
                        )
                    }
                ),
            ]
        )
        _view = mo.ui.tabs({"Overview": _overview, "Topics": _topics})
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Step 4 — Export""")
    return


@app.cell(hide_code=True)
def _(display_result, exports, mo):
    if display_result is None:
        _view = mo.md("*Run the model to download the results.*")
    else:
        _files = {
            "documents_topics.csv": exports.documents_topics_frame(display_result),
            "topics.csv": exports.topics_frame(display_result),
            "topic_terms.csv": exports.topic_terms_frame(display_result),
            "topic_similarity.csv": exports.topic_similarity_frame(display_result),
        }
        _view = mo.hstack(
            [
                mo.download(
                    data=exports.to_csv_bytes(frame),
                    filename=name,
                    label=name,
                    mimetype="text/csv",
                )
                for name, frame in _files.items()
            ],
            justify="start",
            gap=1,
            wrap=True,
        )
    _view
    return


if __name__ == "__main__":
    app.run()
