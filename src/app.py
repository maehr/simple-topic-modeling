import base64
import platform
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.stopwords import german, french, spanish
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from io import StringIO, BytesIO

if not platform.system() == "Emscripten":
    import pyLDAvis
    import pyLDAvis.lda_model

STOP_WORDS = {
    "english": "english",
    "german": german,
    "french": french,
    "spanish": spanish,
}

NGRAM_OPTIONS = {
    "unigram": (1, 1),
    "unigram + bigram": (1, 2),
    "unigram + bigram + trigram": (1, 3),
    "bigram": (2, 2),
    "trigram": (3, 3),
}


def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.read()).decode()


def df_to_base64(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return base64.b64encode(csv_buffer.getvalue().encode()).decode()


def create_topic_df(topic, vectorizer, n_top_words=100):
    normalized_topic = topic / topic.sum()
    top_indices = normalized_topic.argsort()[-n_top_words:]
    top_words = [
        vectorizer.get_feature_names_out()[index] for index in reversed(top_indices)
    ]
    top_probs = [normalized_topic[index] for index in reversed(top_indices)]
    return pd.DataFrame({"Word": top_words, "Probability": top_probs})


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.sidebar.title("About")
    st.sidebar.markdown(
        """
        This app is a simple topic modeling tool that uses Latent Dirichlet Allocation (LDA) to discover topics in a corpus of text documents. 
        It is based on the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
        implementations of LDA and uses [pyLDAvis](https://github.com/bmabey/pyLDAvis) for visualizing the topics.
        """
    )
    st.sidebar.info(
        """
        This app is maintained by [Moritz Mähr](https://maehr.github.io/).
        """,
        icon="ℹ️",
    )

    st.title("Simple Topic Modeling")
    st.markdown(
        """
        Topic modeling is a great way to discover the main themes in a corpus of text documents. 
        It is an unsupervised learning technique that can be used to discover topics in a corpus of documents. 
        Each topic is a distribution over the vocabulary of the corpus. 
        The goal of topic modeling is to find a set of topics that best describes the corpus.
        """
    )

    st.subheader("Step 1: Upload Your Text Files")
    st.markdown(
        """
        Upload your text files containing the documents you want to discover topics for. 
        """
    )

    uploaded_files = st.file_uploader(
        "Upload files containing text data",
        type=["txt", "text", "md", "markdown", "rtf", "csv", "tsv", "log"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        data = [
            {"filename": file.name, "content": file.read().decode("utf-8")}
            for file in uploaded_files
        ]
        df = pd.DataFrame(data)
        st.markdown("**Corpus Statistics**")
        st.write(f"Number of Documents: {df.shape[0]}")
        st.write(
            f"Average Document Length: {df['content'].apply(lambda x: len(x.split())).mean():.2f} words"
        )

        st.subheader("Step 2: Preprocessing the corpus")
        st.text(
            "Choose the preprocessing options that best suit your data. Removing stop words and short words can help improve the quality of the topics generated."
        )

        remove_stop_words = st.checkbox("Remove Stop Words", value=True)
        if remove_stop_words:
            language = st.selectbox("Choose Language for Stop Words", STOP_WORDS.keys())
            use_custom_stop_words = st.checkbox("Use a Custom Stop Words List")
            custom_stop_words = []

        remove_short_words_and_numbers = st.checkbox(
            "Remove Short Words and Numbers", value=True
        )
        if remove_stop_words and use_custom_stop_words:
            custom_stop_words = [
                word.strip()
                for word in st.text_area("Enter Custom Stop Words").split(",")
            ]

        st.markdown(
            """
            Choose the n-gram range for the vectorizer. 
            The n-gram range determines the number of words that are considered as a single token. 
            For example, a unigram range means that each word is considered as a single token. 
            A bigram range means that each pair of words is considered as a single token. 
            A trigram range means that each triplet of words is considered as a single token.
            """
        )
        ngram = st.selectbox("N-Gram Range", list(NGRAM_OPTIONS.keys()))
        ngram_range = NGRAM_OPTIONS[ngram]

        st.subheader("Step 3: Setting the model parameters")
        st.markdown(
            """
            Choose the number of topics and the maximum number of iterations for the model.
            The more iterations, the better the model will fit the data. But it will also take longer to run.
            """
        )
        num_topics = st.slider("Number of Topics", 1, 20, 5)
        max_iter = st.slider("Max Iterations", 10, 500, 50)

        st.subheader("Step 4: Run the topic model and visualize the results")
        st.markdown(
            "Click the button below to run the topic model and discover topics in your corpus. This may take a while depending on the number of documents and the number of topics."
        )
        if st.button("Compute topic model"):
            st.text("Processing...")
            token_pattern = (
                r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b"
                if remove_short_words_and_numbers
                else None
            )
            vectorizer = CountVectorizer(
                lowercase=True,
                stop_words=custom_stop_words
                if use_custom_stop_words
                else STOP_WORDS[language]
                if remove_stop_words
                else None,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
            )
            dtm = vectorizer.fit_transform(df["content"])

            lda = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iter)
            lda_output = lda.fit_transform(dtm)

            dominant_topic = lda_output.argmax(axis=1)
            topic_weights = lda_output.max(axis=1)

            df_topic_weights = pd.DataFrame(
                {"Dominant_Topic": dominant_topic, "Topic_Weight": topic_weights}
            )
            df_topic_weights["Filename"] = df["filename"]

            df_topic_distribution = pd.DataFrame(
                {
                    "Document_Index": df_topic_weights.index,
                    "Filename": df_topic_weights["Filename"],
                    "Dominant_Topic": df_topic_weights["Dominant_Topic"],
                    "Topic_Weight": df_topic_weights["Topic_Weight"],
                }
            )

            df_all_topic_weights = pd.DataFrame(
                lda_output, columns=[i + 1 for i in range(lda_output.shape[1])]
            )
            df_all_topic_weights["Dominant_Topic"] = df_all_topic_weights.idxmax(axis=1)
            df_all_topic_weights_reset = df_all_topic_weights.reset_index(drop=True)
            df_filename_reset = df["filename"].reset_index(drop=True)
            df_topic_distribution = pd.concat(
                [df_filename_reset, df_all_topic_weights_reset], axis=1
            )
            cols = ["filename", "Dominant_Topic"] + [
                col
                for col in df_topic_distribution.columns
                if col not in ["filename", "Dominant_Topic"]
            ]
            df_topic_distribution = df_topic_distribution[cols]

            if not platform.system() == "Emscripten":
                st.subheader("Topics")
                prepared_pyLDAvis_data = pyLDAvis.lda_model.prepare(
                    lda, dtm, vectorizer
                )
                pyLDAvis_html = pyLDAvis.prepared_data_to_html(prepared_pyLDAvis_data)
                st.components.v1.html(
                    pyLDAvis_html, width=1200, height=1000, scrolling=True
                )

            st.subheader("Topic Distribution Over Documents")
            st.dataframe(df_topic_distribution)

            csv_str_base64 = df_to_base64(df_topic_distribution)
            csv_href = f'<a href="data:file/csv;base64,{csv_str_base64}" download="topic_distribution_over_documents.csv">Download Topic Distribution Over Documents</a>'
            st.markdown(csv_href, unsafe_allow_html=True)

            for i, topic in enumerate(lda.components_):
                st.subheader(f"Topic number {i+1}")

                topic_df = create_topic_df(topic, vectorizer, n_top_words=100)

                # Word Cloud
                normalized_topic = topic / topic.sum()
                top_indices = normalized_topic.argsort()[-10:]
                top_words = [
                    vectorizer.get_feature_names_out()[index] for index in top_indices
                ]
                top_probs = [normalized_topic[index] for index in top_indices]

                # Bar Chart
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(top_words, top_probs)
                ax.set_xlabel("Probability")
                ax.set_title(f"Top Words for Topic {i+1}")
                st.pyplot(fig)
                barchart_base64 = fig_to_base64(fig)
                barchart_href = f'<a href="data:image/png;base64,{barchart_base64}" download="barchart_topic_{i+1}.png">Download Bar Chart for Topic {i+1}</a>'
                st.markdown(barchart_href, unsafe_allow_html=True)
                plt.close(fig)

                # Download top 100 words for the topic as CSV
                topic_csv_base64 = df_to_base64(topic_df)
                topic_csv_href = f'<a href="data:file/csv;base64,{topic_csv_base64}" download="top_words_topic_{i+1}.csv">Download Top 100 Words for Topic {i+1}</a>'
                st.markdown(topic_csv_href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
