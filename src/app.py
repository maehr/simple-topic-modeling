import base64
import streamlit as st
import pandas as pd
from utils.stopwords import german, french, spanish
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from io import StringIO
import pyLDAvis
import pyLDAvis.lda_model

# Global variables for stop words and n-gram options
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

def main():
    """
    Main function to run the Simple Topic Modeling app.
    The app allows users to upload a corpus of text documents and discover topics within them using Latent Dirichlet Allocation (LDA).
    It provides options for preprocessing the corpus, setting the model parameters, and visualizing the results.
    """
    # Set page config
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # Sidebar
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

    # Main page
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
    uploaded_files = st.file_uploader(
        "Upload your text files containing the documents you want to discover topics for.",
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
        st.markdown(
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

        if st.button('Compute Topic Model (BE PATIENT)'):
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

            st.subheader("Topics")
            prepared_pyLDAvis_data = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
            pyLDAvis_html = pyLDAvis.prepared_data_to_html(prepared_pyLDAvis_data)
            st.components.v1.html(
                pyLDAvis_html, width=1200, height=800, scrolling=True
            )

            st.subheader("Download Visualization")
            html_buffer = StringIO()
            pyLDAvis.save_html(prepared_pyLDAvis_data, html_buffer)
            html_buffer.seek(0)
            html_str = html_buffer.read()
            html_base64 = base64.b64encode(html_str.encode()).decode()
            html_href = f'<a href="data:text/html;base64,{html_base64}" download="topic_model.html">Download Topic Model</a>'
            st.markdown(html_href, unsafe_allow_html=True)
            
            st.subheader("Download Topic Model")
            json_buffer = StringIO()
            pyLDAvis.save_json(prepared_pyLDAvis_data, json_buffer)
            json_buffer.seek(0)
            json_str = json_buffer.read()
            json_base64 = base64.b64encode(json_str.encode()).decode()
            json_href = f'<a href="data:file/json;base64,{json_base64}" download="topic_model.json">Download Topic Model</a>'
            st.markdown(json_href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
