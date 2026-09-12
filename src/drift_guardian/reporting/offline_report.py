"""Generate a self-contained offline HTML report from a drift report JSON file."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CSS_PATH = Path(__file__).with_name("styles") / "report.css"


def _display(value: Any) -> str:
    """Format and escape a value for safe HTML output."""
    if value is None:
        return "Not specified"
    if isinstance(value, float):
        return f"{value:.3f}"
    return escape(str(value))


def _status(value: Any) -> str:
    """Return a supported status, falling back to an unknown state."""
    status = str(value or "unknown").lower()
    return status if status in {"passed", "warning", "critical"} else "unknown"


def _summary_card(label: str, value: Any, status: str | None = None) -> str:
    status_class = f" status-{_status(status)}" if status else ""
    return f"""
        <article class="summary-card{status_class}">
            <span class="summary-label">{escape(label)}</span>
            <strong class="summary-value">{_display(value)}</strong>
        </article>
    """


def _metrics(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return '<span class="muted">No metrics</span>'
    return "<br>".join(
        f"<span class=\"metric-name\">{escape(name)}</span>: {_display(value)}"
        for name, value in metrics.items()
    )


def _alerts(alerts: list[Any]) -> str:
    if not alerts:
        return '<span class="status-text status-passed-text">No alerts</span>'
    items = "".join(f"<li>{_display(alert)}</li>" for alert in alerts)
    return f'<ul class="alert-list">{items}</ul>'


def _feature_rows(features: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for name, result in features.items():
        status = _status(result.get("status"))
        rows.append(
            f"""
            <tr>
                <th scope="row">{escape(name)}</th>
                <td>{_display(result.get("type"))}</td>
                <td><span class="status-badge status-{status}">{status.upper()}</span></td>
                <td>{_metrics(result.get("metrics", {}))}</td>
                <td>{_alerts(result.get("alerts", []))}</td>
            </tr>
            """
        )
    return "".join(rows)


def render_report_html(report: Mapping[str, Any], css: str) -> str:
    """Render a drift report mapping as a complete HTML document."""
    metadata = report.get("metadata", {})
    features = report.get("features", {})
    prediction = report.get("prediction", {})
    overall_status = _status(report.get("overall_status"))

    summary_cards = "".join(
        [
            _summary_card("Overall status", overall_status.upper(), overall_status),
            _summary_card("Active alerts", report.get("active_alerts", 0)),
            _summary_card("Monitored features", len(features)),
            _summary_card("Window size", report.get("window_size")),
        ]
    )

    metadata_cards = "".join(
        [
            _summary_card("Dataset", metadata.get("dataset_name")),
            _summary_card("Generated at", metadata.get("generated_at")),
            _summary_card("Reference rows", metadata.get("reference_rows")),
            _summary_card("Current rows", metadata.get("current_rows")),
        ]
    )

    prediction_status = _status(prediction.get("status"))

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Data Drift Guardian Report</title>
    <style>
{css}
    </style>
</head>
<body>
    <main class="report-container">
        <header class="report-header">
            <p class="eyebrow">Data Drift Guardian</p>
            <h1>Offline Monitoring Report</h1>
        </header>

        <section aria-labelledby="summary-title">
            <h2 id="summary-title">Overview</h2>
            <div class="card-grid">{summary_cards}</div>
            <div class="card-grid metadata-grid">{metadata_cards}</div>
        </section>

        <section aria-labelledby="features-title">
            <h2 id="features-title">Feature monitoring</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Metrics</th>
                            <th>Alerts</th>
                        </tr>
                    </thead>
                    <tbody>{_feature_rows(features)}</tbody>
                </table>
            </div>
        </section>

        <section aria-labelledby="prediction-title">
            <h2 id="prediction-title">Prediction monitoring</h2>
            <div class="prediction-panel">
                <span class="status-badge status-{prediction_status}">
                    {prediction_status.upper()}
                </span>
                <div>{_metrics(prediction.get("metrics", {}))}</div>
                <div>{_alerts(prediction.get("alerts", []))}</div>
            </div>
        </section>

        <section aria-labelledby="details-title">
            <h2 id="details-title">Feature details</h2>
            <p class="empty-state">
                Interactive Plotly charts will be added when feature distributions
                become available in the drift report.
            </p>
        </section>
    </main>
</body>
</html>
"""


def generate_html_report(
    report: Mapping[str, Any],
    output_path: str | Path,
    css_path: str | Path = DEFAULT_CSS_PATH,
) -> Path:
    output_path = Path(output_path)
    css_path = Path(css_path)

    css = css_path.read_text(encoding="utf-8")
    html = render_report_html(report, css)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def display_html_report(
    report_path: str | Path,
    width: str = "100%",
    height: int = 900,
) -> None:
    """Display a generated HTML report in an isolated Jupyter iframe."""
    from IPython.display import HTML, display

    report_html = Path(report_path).read_text(encoding="utf-8")
    iframe = f"""
    <iframe
        srcdoc="{escape(report_html, quote=True)}"
        width="{escape(width, quote=True)}"
        height="{height}"
        style="border: 1px solid #ddd; border-radius: 8px;"
    ></iframe>
    """
    display(HTML(iframe))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to drift_report.json")
    parser.add_argument("output", type=Path, help="Path to generated HTML")
    args = parser.parse_args()

    output_path = generate_html_report(args.report, args.output)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
