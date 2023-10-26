# simple-topic-modeling

This app is a simple topic modeling tool that uses Latent Dirichlet Allocation (LDA) to discover hidden topics in a corpus of text documents. It's designed for researchers, data scientists, and anyone interested in text analytics. Its basic version runs in the browser.

[![GitHub issues](https://img.shields.io/github/issues/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/issues)
[![GitHub forks](https://img.shields.io/github/forks/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/network)
[![GitHub stars](https://img.shields.io/github/stars/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/stargazers)
[![GitHub license](https://img.shields.io/github/license/maehr/simple-topic-modeling.svg)](https://github.com/maehr/simple-topic-modeling/blob/main/LICENSE.md)

## Demo

You can try out the app [here](https://maehr.github.io/simple-topic-modeling/). If you want to visualize the topics with the more advanced [PyLDAvis](https://github.com/bmabey/pyLDAvis) library, you need to run the app locally.

## Installation

This section provides instructions on how to install the dependencies required to run the app locally. Make sure you have [Python](https://www.python.org/downloads/) 3.11 or higher and [Node.js](https://nodejs.org/en/download/) installed before proceeding.

Use the package manager [poetry](https://python-poetry.org/docs/) to install all Python dependencies. Run the following command in your terminal:

```bash
poetry install
```

Use the package manager [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) to install more dev dependencies like [prettier](https://prettier.io/). Run the following command in your terminal:

```bash
npm install
```

## Usage

To run the app locally, use the following command.

```bash
poetry run streamlit run src/app.py
```

## Development

If you want to contribute to this project, you can use the following commands to test the browser version of the app.

```bash
python3 -m http.server 8000 --directory .
```

To format the code, use the following commands.

```bash
poetry run ruff format
npm format
```

## Support

This project is maintained by [@maehr](https://github.com/maehr). Please understand that we won't be able to provide individual support via email. We also believe that help is much more valuable if it's shared publicly, so that more people can benefit from it.

| Type                                   | Platforms                                                                        |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| 🚨 **Bug Reports**                     | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 📚 **Docs Issue**                      | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 🎁 **Feature Requests**                | [GitHub Issue Tracker](https://github.com/maehr/simple-topic-modeling/issues)    |
| 🛡 **Report a security vulnerability** | See [SECURITY.md](SECURITY.md)                                                   |
| 💬 **General Questions**               | [GitHub Discussions](https://github.com/maehr/simple-topic-modeling/discussions) |

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
