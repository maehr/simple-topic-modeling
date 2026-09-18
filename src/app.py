from io import StringIO

import pandas as pd
import pyLDAvis
import streamlit as st

from topic_model import (
    NGRAM_OPTIONS,
    STOP_WORDS,
    EmptyVocabularyError,
    TooManyTopicsError,
    build_topic_distribution,
    clean_custom_stop_words,
    decode_uploaded_files,
    fit_topic_model,
    prepare_visualization,
    resolve_stop_words,
    resolve_token_pattern,
)


def main():
    """
    Main function to run the Simple Topic Modeling app.
    The app allows users to upload a corpus of text documents and discover
    topics within them using Latent Dirichlet Allocation (LDA).
    It provides options for preprocessing the corpus, setting the model
    parameters, and visualizing the results.
    """
    # Set page config
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown(
        """
        This app is a simple topic modeling tool that uses Latent
        Dirichlet Allocation (LDA) to discover topics in a corpus of
        text documents.
        It is based on the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
        implementations of LDA and uses [pyLDAvis](https://github.com/bmabey/pyLDAvis)
        for visualizing the topics.
        """
    )
    st.sidebar.info(
        """
        This app is maintained by [Moritz Mähr](https://maehr.github.io/).
        """,
        icon="ℹ️",
    )

    # Main page
    st.title("Simple Topic Modeling")
    st.markdown(
        """
        Topic modeling is a great way to discover the main themes in a
        corpus of text documents.
        It is an unsupervised learning technique that can be used to
        discover topics in a corpus of documents.
        Each topic is a distribution over the vocabulary of the corpus.
        The goal of topic modeling is to find a set of topics that best
        describes the corpus.
        """
    )
    st.subheader("Step 1: Upload Your Text Files")
    uploaded_files = st.file_uploader(
        "Upload your text files containing the documents you want to "
        "discover topics for.",
        type=["txt", "text", "md", "markdown", "rtf", "csv", "tsv", "log"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        data, undecodable_filenames = decode_uploaded_files(uploaded_files)
        if undecodable_filenames:
            st.warning(
                "The following files were not valid UTF-8 text. They were "
                "read anyway, with invalid characters replaced: "
                + ", ".join(undecodable_filenames),
                icon="🚨",
            )
        df = pd.DataFrame(data)
        st.markdown("**Corpus Statistics**")
        st.write(f"Number of Documents: {df.shape[0]}")
        average_length = df["content"].apply(lambda x: len(x.split())).mean()
        st.write(f"Average Document Length: {average_length:.2f} words")
        st.subheader("Step 2: Preprocessing the corpus")
        st.markdown(
            "Choose the preprocessing options that best suit your data. "
            "Removing stop words and short words can help improve the "
            "quality of the topics generated."
        )
        remove_stop_words = st.checkbox("Remove Stop Words", value=True)
        language = "english"
        use_custom_stop_words = False
        custom_stop_words = []
        if remove_stop_words:
            language = st.selectbox("Choose Language for Stop Words", STOP_WORDS.keys())
            use_custom_stop_words = st.checkbox("Use a Custom Stop Words List")
            if use_custom_stop_words:
                custom_stop_words = clean_custom_stop_words(
                    st.text_area("Enter Custom Stop Words separated by a comma")
                )
        remove_short_words_and_numbers = st.checkbox(
            "Remove Short Words and Numbers", value=True
        )
        st.markdown(
            """
            Choose the n-gram range for the vectorizer.
            The n-gram range determines the number of words that are
            considered as a single token.
            For example, a unigram range means that each word is
            considered as a single token.
            A bigram range means that each pair of words is considered
            as a single token.
            A trigram range means that each triplet of words is
            considered as a single token.
            """
        )
        ngram = st.selectbox("N-Gram Range", list(NGRAM_OPTIONS.keys()))
        ngram_range = NGRAM_OPTIONS[ngram]
        st.subheader("Step 3: Setting the model parameters")
        st.markdown(
            """
            Choose the number of topics and the maximum number of
            iterations for the model.
            The more iterations, the better the model will fit the data.
            But it will also take longer to run.
            """
        )
        num_topics = st.slider("Number of Topics", 1, 20, 5)
        max_iter = st.slider("Max Iterations", 10, 500, 50)
        st.subheader("Step 4: Run the topic model and visualize the results")
        st.markdown(
            "Click the button below to run the topic model and discover "
            "topics in your corpus. This may take a while depending on "
            "the number of documents and the number of topics."
        )
        # The results are kept in st.session_state, not rendered straight from
        # the button branch. st.button is True only on the rerun its own click
        # causes, and st.download_button triggers a rerun too. Rendering inside
        # the button branch therefore made the whole results section disappear
        # as soon as the user downloaded any one of the three files.
        stop_words = resolve_stop_words(
            remove_stop_words, use_custom_stop_words, custom_stop_words, language
        )
        stop_words_arg = (
            tuple(stop_words) if isinstance(stop_words, list) else stop_words
        )
        token_pattern = resolve_token_pattern(remove_short_words_and_numbers)

        # Stored results belong to one exact set of inputs. Anything else would
        # show a stale model after the user changes a setting.
        signature = (
            tuple(df["filename"]),
            stop_words_arg,
            token_pattern,
            ngram_range,
            num_topics,
            max_iter,
        )

        if st.button("Compute Topic Model"):
            st.session_state.pop("results", None)
            with st.status("Computing topic model...", expanded=True) as status:
                try:
                    st.write("Vectorizing corpus and fitting LDA model...")
                    vectorizer, dtm, lda, lda_output = fit_topic_model(
                        tuple(df["content"]),
                        stop_words_arg,
                        token_pattern,
                        ngram_range,
                        num_topics,
                        max_iter,
                    )
                except (EmptyVocabularyError, TooManyTopicsError) as exc:
                    status.update(label="Failed", state="error")
                    st.error(str(exc), icon="\U0001f6a8")
                    return

                st.write("Preparing visualization...")
                prepared = prepare_visualization(lda, dtm, vectorizer)
                df_topic_distribution = build_topic_distribution(
                    df["filename"], lda_output, prepared.topic_order
                )
                status.update(label="Done!", state="complete")

            html_buffer = StringIO()
            pyLDAvis.save_html(prepared, html_buffer)
            json_buffer = StringIO()
            pyLDAvis.save_json(prepared, json_buffer)
            csv_buffer = StringIO()
            df_topic_distribution.to_csv(csv_buffer, index=False)

            st.session_state["results"] = {
                "signature": signature,
                "figure": pyLDAvis.prepared_data_to_html(prepared),
                "html": html_buffer.getvalue(),
                "json": json_buffer.getvalue(),
                "csv": csv_buffer.getvalue(),
            }

        results = st.session_state.get("results")
        if results and results["signature"] == signature:
            st.subheader("Topics")
            st.components.v1.html(
                results["figure"], width=1200, height=800, scrolling=True
            )

            st.subheader("Download Visualization")
            st.download_button(
                "Download Visualization",
                data=results["html"],
                file_name="topic_model.html",
                mime="text/html",
            )

            st.subheader("Download Topic Model")
            st.download_button(
                "Download Topic Model",
                data=results["json"],
                file_name="topic_model.json",
                mime="application/json",
            )

            st.subheader("Download Topic Distribution")
            st.download_button(
                "Download Topic Distribution",
                data=results["csv"],
                file_name="topic_distribution.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
