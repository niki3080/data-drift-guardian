from config.parse_config import Config, FeatureType

from typing import Optional, List, Tuple

def extract_feature_groups(
    config: Config,
) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str]]:
    """
    Извлекает из конфига:
      - num_features: список имён numeric-фичей (или None, если таких нет)
      - cat_features: список имён categorical-фичей (или None, если таких нет)
      - prediction: score_column, если prediction_metrics.enabled=True (иначе None)
    """
    num_features = [
        name
        for name, feature in config.features.items()
        if feature.type == FeatureType.numeric
    ]
    cat_features = [
        name
        for name, feature in config.features.items()
        if feature.type == FeatureType.categorical
    ]

    prediction = (
        config.prediction_metrics.score_column
        if config.prediction_metrics.enabled
        else None
    )

    return (
        num_features or None,
        cat_features or None,
        prediction,
    )
