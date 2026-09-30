# Plan — second demo (#42) and About sidebar (#43)

## Stack (each PR targets the branch below it)

| Issue | Branch | Base |
|-------|--------|------|
| #42 | `feat/demo-text` | `main` |
| #43 | `feat/about-sidebar` | `feat/demo-text` |

Merge from the bottom up.

## #42 — second demo: one long text

Decisions:

- Source: Project Gutenberg #3420, `https://www.gutenberg.org/cache/epub/3420/pg3420.txt`.
  The script pins the SHA-256. A mismatch stops the build. The maintainer then reviews the new
  file and updates the hash. No mirror: the pinned hash already detects a change in place.
- `scripts/build_demo_text.py` writes `simple_topic_modeling/data/demo_text.txt` and
  `scripts/demo_text.provenance.json`. It removes the Gutenberg header, footer, and licence.
  It joins each hard-wrapped paragraph into one line and keeps a blank line between paragraphs.
- `io.demo_text()` reads the file with `importlib.resources`.
- The source radio gets two demo options: **Demo: newspapers** and **Demo: book**.
- Two buttons above Step 1: **Run demo** (newspapers) and **Run book demo**.
- The book demo opens in **Ordered text** mode, split on blank lines, in English.
- The language controls join the `touched` map, so a change of source keeps the reader's values.
- The start topic count for the book comes from a measured test. The comment in `app.py`
  holds the reason.
- Update `NOTICE`, the About section, the language help, and `SPECS.md`.

## #43 — About in the sidebar

- Move **Run**, the summary, and **Run model** to the start of Step 3. Keep `run_button`.
- Put the About text in `mo.sidebar`. Remove the About cell at the end of the page.
- Update `SPECS.md` Step 3 and the About rule.

## Checks for each PR

- The section 4 gate, `marimo check`, and the `app.run()` check.
- Every cell parameter is some cell's return value.
- Browser: Run model, Run demo, Run book demo, and Rerun each start one fit, in `marimo edit`
  and in the exported `dist/`.
- #43: the sidebar at phone width.

## Status

- [x] #42 build script
- [x] #42 package and app
- [x] #42 checks (gate, app.run, marimo edit + dist browser runs)
- [ ] #43 app and specs
- [ ] #43 checks, PR
