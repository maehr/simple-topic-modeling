"""Run the notebook outside the browser.

`marimo check` does not find a name that two cells both define, and a plain `app.run()` takes only
the path without a result. These tests close both gaps: every cell parameter must come from a
cell, and the result view must run with a fitted result.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import TypeGuard

import pytest

from simple_topic_modeling.config import AppConfig, ModelConfig
from simple_topic_modeling.io import build_corpus, demo_table, demo_text, split_long_document
from simple_topic_modeling.modeling import fit_topic_model

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _is_cell(node: ast.AST) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == "cell":
            return True
    return False


def _returned_names(cell: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for statement in cell.body:
        if isinstance(statement, ast.Return) and statement.value is not None:
            values = statement.value
            items = values.elts if isinstance(values, ast.Tuple) else [values]
            names.update(item.id for item in items if isinstance(item, ast.Name))
    return names


@pytest.fixture(scope="module")
def notebook():
    spec = importlib.util.spec_from_file_location("notebook_app", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["notebook_app"] = module
    spec.loader.exec_module(module)
    return module


def test_every_cell_parameter_is_returned_by_a_cell():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    cells = [node for node in ast.walk(tree) if _is_cell(node)]
    returned = set().union(*(_returned_names(cell) for cell in cells))
    missing = {
        f"{cell.lineno}:{argument.arg}"
        for cell in cells
        for argument in cell.args.args
        if argument.arg not in returned
    }
    assert not missing, f"Cell parameters that no cell returns: {sorted(missing)}"


def test_the_notebook_runs_without_a_result(notebook):
    _, definitions = notebook.app.run()
    assert definitions["display_result"] is None


def test_the_result_view_runs_with_the_demo_corpus(notebook):
    table = demo_table()
    metadata = table[["category", "date"]].rename(columns={"category": "group"})
    corpus, _ = build_corpus(
        table["text"].tolist(), table["document_id"].astype(str).tolist(), metadata
    )
    result = fit_topic_model(corpus, AppConfig())
    _, definitions = notebook.app.run(defs={"display_result": result})
    assert definitions["map_chart"] is not None
    assert definitions["selected_index"] == 0


def test_the_result_view_runs_with_the_long_text_demo(notebook):
    segments, identifiers, metadata = split_long_document(demo_text(), "demo.txt")
    corpus, _ = build_corpus(segments, identifiers, metadata)
    config = AppConfig(model=ModelConfig(n_topics=4), analyse_as="long_document")
    result = fit_topic_model(corpus, config)
    _, definitions = notebook.app.run(defs={"display_result": result})
    assert definitions["map_chart"] is not None
