# simple-topic-modeling

This app is a simple topic modeling tool that uses Latent Dirichlet Allocation (LDA) to discover hidden topics in a corpus of text documents. It's designed for researchers, data scientists, and anyone interested in text analytics. Its basic version runs in the browser.

[![GitHub issues](https://img.shields.io/github/issues/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/issues)
[![GitHub forks](https://img.shields.io/github/forks/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/network)
[![GitHub stars](https://img.shields.io/github/stars/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/stargazers)
[![GitHub license](https://img.shields.io/github/license/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/blob/main/LICENSE.md)

## Demo

You can try out the app [here](https://maehr.github.io/simple-topic-modeling/).

## Installation

This section provides instructions on how to install the dependencies required to run the app locally. Make sure you have [Python](https://www.python.org/downloads/) 3.11 or higher and [Node.js](https://nodejs.org/en/download/) installed before proceeding.

Use the package manager [poetry](https://python-poetry.org/docs/) to install all Python dependencies. Run the following command in your terminal:

```bash
poetry install
```

Use the package manager [pnpm](https://pnpm.io/installation) to install dev dependencies like [prettier](https://prettier.io/). Run the following command in your terminal:

```bash
pnpm install
```

## Usage

To run the app locally, use the following command.

```bash
poetry run streamlit run src/app.py
```

## Development

To run linting and formatting checks, use these commands:

```bash
poetry run ruff check .
poetry run ruff format .
pnpm check
pnpm format
```

To run the tests, use this command:

```bash
poetry run pytest
```

### Browser version

The app runs in the browser through stlite (Streamlit compiled to WebAssembly). Before testing locally, download the runtime:

```bash
python3 scripts/fetch_stlite.py
```

This script downloads the pinned stlite release into `assets/stlite/build/` and verifies it against the hash in `scripts/stlite.lock.json`.

Serve the repository root at your development machine:

```bash
python3 -m http.server 8000
```

Open the directory URL `http://localhost:8000/` in your browser, not the file URL. stlite builds its internal asset URLs from the page location, and the file URL form breaks them.

To update to a new stlite version, run:

```bash
python3 scripts/fetch_stlite.py --update <version>
```

### Python wheel

The pyLDAvis wheel is committed to `assets/dist/pyLDAvis-3.4.1-py3-none-any.whl`. Rebuild it reproducibly with:

```bash
python3 scripts/build_pyldavis_wheel.py
```

This script downloads the upstream wheel, verifies its SHA-256, removes pip and setuptools that upstream accidentally ships inside, and removes gensim and numexpr requirements that cannot run in Pyodide. Verify the output hash with:

```bash
shasum -c scripts/pyldavis-wheel.sha256
```

### Dependencies

`pyproject.toml` caps numpy below 2.3. pyLDAvis 3.4.1 cannot serialize values from newer numpy versions, causing the app to fail at the visualization step. This cap must remain until pyLDAvis fixes it.

## Support

This project is maintained by [@maehr](https://github.com/maehr). Please understand that we won't be able to provide individual support via email. We also believe that help is much more valuable if it's shared publicly, so that more people can benefit from it.

| Type                                  | Platforms                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| 🚨 **Bug Reports**                    | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 📚 **Docs Issue**                     | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 🎁 **Feature Requests**               | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 🛡 **Report a security vulnerability** | See [SECURITY.md](SECURITY.md)                                                   |
| 💬 **General Questions**              | [GitHub Discussions](https://github.com/maehr/simple-topic-modeling/discussions) |

## Roadmap

No changes are currently planned.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/maehr/simple-topic-modeling/tags).

## Authors and acknowledgment

- **Moritz Mähr** - _Initial work_ - [maehr](https://github.com/maehr)

See also the list of [contributors](https://github.com/maehr/simple-topic-modeling/graphs/contributors) who participated in this project.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.
