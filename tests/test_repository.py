import ast
import io
import json
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTest(unittest.TestCase):
    def test_main(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main.main()
        self.assertEqual(output.getvalue().strip(), "Hello from drift-guardian!")

    def test_pyproject(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)

        self.assertEqual(project["project"]["name"], "drift-guardian")
        self.assertEqual(project["project"]["requires-python"], ">=3.13")

    def test_notebook_python_cells(self) -> None:
        path = ROOT / "notebooks" / "01_library_research.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            if source.strip():
                ast.parse(source, filename=f"notebook-cell-{index}")

    def test_sample_parquet(self) -> None:
        path = ROOT / "data" / "train_transaction_sample.parquet"
        content = path.read_bytes()

        self.assertGreater(len(content), 8)
        self.assertEqual(content[:4], b"PAR1")
        self.assertEqual(content[-4:], b"PAR1")

    def test_research_reports(self) -> None:
        reports = ROOT / "notebooks" / "reports"
        expected = [
            "evidently_data_drift.html",
            "evidently_datasummary.html",
            "evidently_datasummary_auto.html",
            "nannyml_univariate_drift.html",
        ]

        for name in expected:
            with self.subTest(name=name):
                path = reports / name
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
