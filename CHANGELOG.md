# Changelog

This file records every notable change to Simple Topic Modeling.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version 2.0.0 replaces version 1, which was a Streamlit app that ran through stlite. The two share
no code. Version 1 stays at the tag `v1.0.0` and the branch `legacy/streamlit`, and it receives no
further work.

## [2.0.0] - 2026-09-20

### Added

- Add corpus import, cleaning and stop-word modules
- Add the topic model, the result object and the diagnostics
- Add the CSV, JSON and ZIP exports
- Add the marimo app, the demo corpus and the chart data layer
- Add the topic map, heatmap, word cloud and document explorer
- Add the metadata and diagnostics tabs and the ZIP export
- Warn about a large corpus and offer sampling
- **Breaking:** Release as version 2.0.0

### Build and tooling

- Scaffold project, tooling gate and WASM export path
- Add the workflows, the health files and Dependabot

### Documentation

- Expand the README and record the project decisions
- Record the move to the remote and the version 1 retirement

