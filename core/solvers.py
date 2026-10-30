"""Собственная реализация алгоритмов регрессии на numpy.

Все три алгоритма ищут модель ŷ = b0 + b1·x1 + ... + bp·xp.
Свободный член b0 не штрафуется: данные центрируются (из X и y вычитаются
средние), коэффициенты ищутся по центрированным данным, затем
b0 = ȳ - Σ bᵢ·x̄ᵢ.

Целевые функции совпадают с принятыми в scikit-learn, поэтому результаты
можно сверять с этой библиотекой (см. check_core.py):
  линейная:  Σ(y - ŷ)²                                → min
  Ridge:     Σ(y - ŷ)² + α·Σb²                        → min
  Lasso:     (1 / 2n)·Σ(y - ŷ)² + α·Σ|b|              → min
"""

import numpy as np


class LeastSquaresEstimator:
    """Решатель задачи регрессии. penalty: None | "l2" (Ridge) | "l1" (Lasso)."""

    def __init__(self, penalty: str | None = None, alpha: float = 1.0,
                 max_iter: int = 10_000, tol: float = 1e-6):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter   # максимум итераций (только для Lasso)
        self.tol = tol             # точность остановки (только для Lasso)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_iter_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean            # центрирование

        if self.penalty is None:
            # МНК: решение системы методом наименьших квадратов
            w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        elif self.penalty == "l2":
            # Ridge: (XᵀX + αI)·w = Xᵀy — аналитическое решение
            p = X.shape[1]
            w = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(p), Xc.T @ yc)
        elif self.penalty == "l1":
            w = self._lasso_coordinate_descent(Xc, yc)
        else:
            raise ValueError(f"Неизвестный штраф: {self.penalty}")

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def _lasso_coordinate_descent(self, X, y):
        """Lasso методом покоординатного спуска.

        На каждом шаге один коэффициент bⱼ находится точно при остальных
        фиксированных, с «мягким порогом»:
            ρⱼ = xⱼᵀ·(rⱼ),  bⱼ = sign(ρⱼ)·max(|ρⱼ| - α·n, 0) / ‖xⱼ‖²,
        где rⱼ — остатки без вклада j-го признака. Поэтому Lasso может
        обнулять коэффициенты неинформативных признаков.
        """
        n, p = X.shape
        w = np.zeros(p)
        r = y.copy()                        # остатки y - X·w
        col_sq = (X ** 2).sum(axis=0)       # ‖xⱼ‖²
        threshold = self.alpha * n
        for it in range(1, self.max_iter + 1):
            max_change, max_w = 0.0, 0.0
            for j in range(p):
                if col_sq[j] == 0:
                    continue                # постоянный признак
                old = w[j]
                rho = X[:, j] @ r + col_sq[j] * old
                new = np.sign(rho) * max(abs(rho) - threshold, 0.0) / col_sq[j]
                if new != old:
                    r -= X[:, j] * (new - old)
                    w[j] = new
                max_change = max(max_change, abs(new - old))
                max_w = max(max_w, abs(new))
            self.n_iter_ = it
            if max_w == 0.0 or max_change / max_w < self.tol:
                break                       # коэффициенты перестали меняться
        return w

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
