"""Generate a standalone HTML report from an offline drift-report."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CSS_PATH = Path(__file__).with_name("styles") / "report.css"

STATUS_LABELS = {
    "passed": "OK",
    "warning": "WARNING",
    "critical": "CRITICAL",
    "unknown": "UNKNOWN",
}

OVERALL_DETAILS = {
    "passed": "No input data drift detected",
    "warning": "Input data drift warning detected",
    "critical": "Input data drift detected",
    "unknown": "Input data drift status is unavailable",
}

AV_WARNING_THRESHOLD = 0.60
AV_CRITICAL_THRESHOLD = 0.75

AV_DETAILS = {
    "passed": "No adversarial drift detected",
    "warning": "Adversarial drift warning detected",
    "critical": "Adversarial drift detected",
    "unknown": "Adversarial validation status is unavailable",
}


def _status(value: Any) -> str:
    """Normalize engine statuses to statuses supported by the report UI."""
    status = str(value or "unknown").lower()
    if status == "ok":
        return "passed"
    return status if status in {"passed", "warning", "critical"} else "unknown"


def _display(value: Any, *, decimals: int = 3) -> str:
    """Format and escape a value for HTML output."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return escape(str(value))
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:.{decimals}f}"
    return escape(str(value))


def _metric_label(name: str) -> str:
    """Turn a metric key into a compact human-readable table heading."""
    labels = {
        "cardinality_ratio": "Card. ratio",
        "js_divergence": "JS div.",
        "kstest": "KS test",
        "missing_rate": "Missing rate",
        "psi": "PSI",
        "unseen_category_rate": "Unseen rate",
        "wasserstein": "Wasserstein",
        "wasserstein_distance": "Wasserstein",
        "chi2": "χ²-test",
        "cramer_v": "Cramér's V",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def _summary_card(
    label: str,
    value: Any,
    *,
    status: str | None = None,
    details: str = "",
    variant: str = "alert",
) -> str:
    """Render one status/metric card."""
    status_class = f" status-{_status(status)}" if status else ""
    return f"""
        <article class="summary-card summary-card-{escape(variant)}{status_class}">
            <span class="summary-label">{escape(label)}</span>
            <strong class="summary-value">{_display(value)}</strong>
            <span class="summary-details">{escape(details)}</span>
        </article>
    """


def _roc_auc_card(value: Any, status: str) -> str:
    """Render ROC AUC with the same threshold markers as metric cells."""
    thresholds = "".join(
        [
            _threshold(AV_WARNING_THRESHOLD, "warning"),
            _threshold(AV_CRITICAL_THRESHOLD, "critical"),
        ]
    )
    return f"""
        <article class="summary-card summary-card-alert">
            <span class="summary-label">ROC AUC</span>
            <strong class="summary-value status-text-{status}">{_display(value)}</strong>
            <div class="metric-thresholds">{thresholds}</div>
        </article>
    """


def _status_dot(status: Any) -> str:
    normalized = _status(status)
    return (
        f'<span class="status-dot status-dot-{normalized}" '
        f'title="{STATUS_LABELS[normalized]}"></span>'
    )


def _threshold(value: Any, level: str) -> str:
    if value is None:
        return ""
    return (
        f'<span class="threshold threshold-{level}">'
        f'<span class="threshold-marker"></span>{_display(value)}</span>'
    )


def _metric_cell(metric: Any) -> str:
    if metric is None:
        return '<span class="muted">—</span>'

    # Older saved reports contain only a scalar metric value. They have no
    # per-metric status or thresholds to colour, but remain renderable.
    if not isinstance(metric, Mapping):
        return f'<div class="metric-value">{_display(metric)}</div>'

    status = _status(metric.get("status"))
    thresholds = "".join(
        [
            _threshold(metric.get("warning"), "warning"),
            _threshold(metric.get("critical"), "critical"),
        ]
    )
    return f"""
        <div class="metric-value status-text-{status}">
            {_display(metric.get("value"))}
        </div>
        <div class="metric-thresholds">{thresholds}</div>
    """


def _metric_names(features: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return metric columns in their first-seen order."""
    names: list[str] = []
    for result in features.values():
        for metric_name in result.get("metrics", {}):
            if metric_name not in names:
                names.append(metric_name)
    if "psi" in names:
        names.remove("psi")
        names.insert(0, "psi")
    return names


def _feature_table(
    title: str,
    features: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render a metric-column table for one feature type."""
    if not features:
        return ""

    metric_names = _metric_names(features)
    metric_headers = "".join(
        f'<th scope="col">{escape(_metric_label(name))}</th>'
        for name in metric_names
    )
    heading = f"<h3>{escape(title)}</h3>" if title else ""
    rows = []
    for feature_name, result in features.items():
        metrics = result.get("metrics", {})
        metric_cells = "".join(
            f"<td>{_metric_cell(metrics.get(name))}</td>" for name in metric_names
        )
        rows.append(
            f"""
            <tr>
                <td class="status-column">{_status_dot(result.get("status"))}</td>
                <th class="feature-name" scope="row">{escape(str(feature_name))}</th>
                {metric_cells}
            </tr>
            """
        )

    return f"""
        <section class="feature-group" aria-label="{escape(title or 'Feature metrics')}">
            {heading}
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th class="status-column" scope="col">Status</th>
                            <th scope="col">Feature</th>
                            {metric_headers}
                        </tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </section>
    """


def _count_statuses(features: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int, int]:
    warning_features = 0
    critical_features = 0
    warning_metrics = 0
    critical_metrics = 0

    for result in features.values():
        feature_status = _status(result.get("status"))
        warning_features += feature_status == "warning"
        critical_features += feature_status == "critical"
        for metric in result.get("metrics", {}).values():
            if not isinstance(metric, Mapping):
                continue
            metric_status = _status(metric.get("status"))
            warning_metrics += metric_status == "warning"
            critical_metrics += metric_status == "critical"

    return warning_features, critical_features, warning_metrics, critical_metrics


def _feature_list(features: Mapping[str, Mapping[str, Any]]) -> str:
    groups: dict[str, list[str]] = {"numeric": [], "categorical": []}
    for name, result in features.items():
        feature_type = str(result.get("type", "unknown")).lower()
        groups.setdefault(feature_type, []).append(str(name))

    items = "".join(
        f"<li><b>{escape(feature_type)}:</b> {escape(', '.join(names))}</li>"
        for feature_type, names in groups.items()
        if names
    )
    return f'<ul class="feature-list">{items}</ul>'


def _av_status(roc_auc: Any) -> str:
    if not isinstance(roc_auc, Real) or not math.isfinite(float(roc_auc)):
        return "unknown"
    if float(roc_auc) >= AV_CRITICAL_THRESHOLD:
        return "critical"
    if float(roc_auc) >= AV_WARNING_THRESHOLD:
        return "warning"
    return "passed"


def _feature_importance_table(importance: Any) -> str:
    required_columns = {"rank", "feature", "importance"}
    columns = set(getattr(importance, "columns", []))
    if not required_columns.issubset(columns):
        return '<p class="muted">Feature importance is unavailable</p>'

    table_columns = ["rank", "feature", "importance"]
    if "importance_std" in columns:
        table_columns.append("importance_std")

    rows = importance.head(10).loc[:, table_columns].to_dict(orient="records")
    headers = {
        "rank": "Rank",
        "feature": "Feature",
        "importance": "Importance",
        "importance_std": "Importance std.",
    }
    header_html = "".join(
        f'<th scope="col">{headers[column]}</th>' for column in table_columns
    )
    row_html = "".join(
        "<tr>"
        + "".join(f"<td>{_display(row.get(column))}</td>" for column in table_columns)
        + "</tr>"
        for row in rows
    )
    return f"""
        <div class="table-wrapper">
            <table class="feature-importance-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{row_html}</tbody>
            </table>
        </div>
    """


def _adversarial_validation_section(av_report: Any) -> str:
    if av_report is None:
        return ""
    if not isinstance(av_report, (tuple, list)) or len(av_report) != 2:
        raise TypeError("av_report must be the (roc_auc, feature_importance) result of run_av")

    roc_auc, importance = av_report
    status = _av_status(roc_auc)
    cards = "".join(
        [
            _summary_card(
                "Adversarial validation",
                STATUS_LABELS[status],
                status=status,
                details=AV_DETAILS[status],
                variant="overall",
            ),
            _roc_auc_card(roc_auc, status),
        ]
    )
    return f"""
        <section class="report-section" aria-labelledby="av-title">
            <h2 id="av-title">Adversarial Validation</h2>
            <div class="card-grid">{cards}</div>
            <section class="feature-group" aria-label="Feature importance">
                <h3>feature importance (top 10)</h3>
                {_feature_importance_table(importance)}
            </section>
        </section>
    """


def render_report_html(
    report: Mapping[str, Any],
    css: str,
    *,
    dataset_name: str = "",
    av_report: Any = None,
) -> str:
    """Build a complete responsive HTML document from a drift report."""
    metadata = report.get("metadata", {})
    features = report.get("features", {})
    prediction = report.get("prediction")
    overall_status = _status(report.get("overall_status"))
    evaluated = len(features)
    warning_features, counted_critical, warning_metrics, critical_metrics = (
        _count_statuses(features)
    )
    critical_features = report.get("active_alerts", counted_critical)
    report_dataset_name = dataset_name or str(metadata.get("dataset_name") or "Dataset")

    summary_cards = "".join(
        [
            _summary_card(
                "Overall status",
                STATUS_LABELS[overall_status],
                status=overall_status,
                details=OVERALL_DETAILS[overall_status],
                variant="overall",
            ),
            _summary_card(
                "Critical \nfeatures",
                critical_features,
                details=f"of {evaluated} evaluated",
            ),
            _summary_card(
                "Warning \nfeatures",
                warning_features,
                details=f"of {evaluated} evaluated",
            ),
            _summary_card("Warning \nmetric alerts", warning_metrics),
            _summary_card("Critical \nmetric alerts", critical_metrics),
        ]
    )

    grouped_features: dict[str, dict[str, Mapping[str, Any]]] = {}
    for feature_name, result in features.items():
        feature_type = str(result.get("type", "unknown")).lower()
        grouped_features.setdefault(feature_type, {})[str(feature_name)] = result

    feature_tables = "".join(
        _feature_table(f"{feature_type} features", typed_features)
        for feature_type, typed_features in grouped_features.items()
    )

    prediction_html = ""
    if isinstance(prediction, Mapping) and prediction:
        prediction_html = f"""
        <section class="report-section" aria-labelledby="prediction-title">
            <h2 id="prediction-title">Prediction Drift</h2>
            {_feature_table("", {"prediction": prediction})}
        </section>
        """

    adversarial_validation_html = _adversarial_validation_section(av_report)

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(report_dataset_name)} — Data Drift Guardian</title>
    <style>
{css}
    </style>
</head>
<body>
    <main class="report-container">
        <header class="report-header">
            <h1>{escape(report_dataset_name)}</h1>
            <div class="report-context">
                <span>Report timestamp: <b>{_display(report.get('timestamp'))}</b></span>
                <span>Window size: <b>{_display(report.get('window_size'))}</b></span>
            </div>
            <div class="monitoring-features">
                <b>Monitored features:</b>
                {_feature_list(features)}
            </div>
        </header>

        <section class="report-section" aria-labelledby="input-drift-title">
            <h2 id="input-drift-title">Input Data Drift</h2>
            <div class="card-grid">{summary_cards}</div>
        </section>

        <section class="report-section" aria-labelledby="feature-drift-title">
            <h2 id="feature-drift-title">Feature-level Drift</h2>
            {feature_tables or '<p class="muted">No evaluated features</p>'}
        </section>

        {adversarial_validation_html}

        {prediction_html}
    </main>
</body>
</html>
"""


def generate_html_report(
    report: Mapping[str, Any] | str | Path | None = None,
    *,
    av_report: Any = None,
    dataset_name: str = "",
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    css_path: str | Path = DEFAULT_CSS_PATH,
) -> Path:
    """Generate an HTML report from a mapping or a JSON file."""
    if report_path is not None:
        if report is not None:
            raise ValueError("provide report or report_path, not both")
        report = report_path
    if report is None:
        raise ValueError("report or report_path is required")
    if output_path is None:
        raise ValueError("output_path is required")

    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping or a path to JSON")

    output_path = Path(output_path)
    css = Path(css_path).read_text(encoding="utf-8")
    html = render_report_html(
        report,
        css,
        dataset_name=dataset_name,
        av_report=av_report,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def display_html_report(
    report_path: str | Path,
    width: str = "100%",
    height: int | str = 1200,
) -> None:
    """Display a generated report in an isolated Jupyter iframe."""
    from IPython.display import HTML, display

    report_html = Path(report_path).read_text(encoding="utf-8")
    iframe = f"""
    <iframe
        srcdoc="{escape(report_html, quote=True)}"
        width="{escape(width, quote=True)}"
        height="{height}"
        style="border: 0; border-radius: 8px;"
    ></iframe>
    """
    display(HTML(iframe))


def main() -> None:
    """CLI entry point for generating an offline HTML report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to drift_report.json")
    parser.add_argument("output", type=Path, help="Path to generated HTML")
    parser.add_argument("--dataset-name", default="", help="Dataset name in the report header")
    args = parser.parse_args()

    output_path = generate_html_report(
        args.report,
        args.output,
        dataset_name=args.dataset_name,
    )
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
