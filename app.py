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

app = marimo.App(width="medium", app_title="Browser Topic Explorer")


@app.cell
def _():
    import marimo as mo

    import browser_topics

    mo.md(
        f"""
        # Browser Topic Explorer

        Version `{browser_topics.__version__}`. The user interface arrives in phase 4.
        """
    )
    return


if __name__ == "__main__":
    app.run()
