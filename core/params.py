"""Класс ModelParams — параметры моделирования (п. 4.1.2.3 ТЗ)."""

from dataclasses import dataclass, field

# Названия алгоритмов для интерфейса и отчётов
ALGORITHM_NAMES = {
    "linear": "Линейная регрессия",
    "ridge": "Гребневая регрессия (Ridge)",
    "lasso": "Лассо-регрессия (Lasso)",
}


@dataclass
class ModelParams:
    algorithm: str = "linear"          # linear | ridge | lasso
    target: str = ""                   # целевой признак
    features: list[str] = field(default_factory=list)  # входные признаки
    alpha: float = 1.0                 # α для Ridge и Lasso, > 0
    standardize: bool = True           # стандартизация входных признаков
    test_size: float = 0.2             # доля тестовой выборки, 0.10–0.40
    seed: int = 42                     # начальное значение генератора
    missing_method: str = "drop"       # drop | mean
