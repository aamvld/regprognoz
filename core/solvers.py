"""Собственная реализация алгоритмов регрессии на numpy.

Все три алгоритма ищут модель ŷ = b0 + b1·x1 + ... + bp·xp.
Свободный член b0 не штрафуется: данные центрируются (из X и y вычитаются
средние), коэффициенты ищутся по центрированным данным, затем
b0 = ȳ - Σ b_i·x̄_i.

Целевые функции совпадают с принятыми в scikit-learn, поэтому результаты
можно сверять с этой библиотекой (см. check_core.py):
  линейная:  Σ(y - ŷ)²                                → min
  Ridge:     Σ(y - ŷ)² + α·Σb²                        → min
  Lasso:     (1 / 2n)·Σ(y - ŷ)² + α·Σ|b|              → min
"""

import numpy as np


class LeastSquaresEstimator:
    """Решатель задачи регрессии.

    penalty: None – МНК, "l2" – Ridge, "l1" – Lasso.
    """

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
            # Ridge: (X^TX + αI)·w = X^Ty – аналитическое решение
            p = X.shape[1]
            A = Xc.T @ Xc + self.alpha * np.eye(p)
            w = np.linalg.solve(A, Xc.T @ yc)
        elif self.penalty == "l1":
            w = self._lasso_coordinate_descent(Xc, yc)
        else:
            raise ValueError(f"Неизвестный штраф: {self.penalty}")

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def _lasso_coordinate_descent(self, X, y):
        """Lasso методом покоординатного спуска.

        На каждом шаге один коэффициент b_j находится точно при остальных
        фиксированных, с «мягким порогом»:
            ρ_j = x_j^T·(r_j),
            b_j = sign(ρ_j)·max(|ρ_j| - α·n, 0) / ||x_j||^2,
        где r_j – остатки без вклада j-го признака. Поэтому Lasso может
        обнулять коэффициенты неинформативных признаков.
        """
        n, p = X.shape
        w = np.zeros(p)
        r = y.copy()                        # остатки y - X·w
        col_sq = (X ** 2).sum(axis=0)       # ||x_j||^2
        threshold = self.alpha * n
        for it in range(1, self.max_iter + 1):
            max_change, max_w = 0.0, 0.0
            for j in range(p):
                if col_sq[j] == 0:
                    continue                # постоянный признак
                old = w[j]
                rho = X[:, j] @ r + col_sq[j] * old
                shrunk = max(abs(rho) - threshold, 0.0)   # «мягкий порог»
                new = np.sign(rho) * shrunk / col_sq[j]
                if new != old:
                    r -= X[:, j] * (new - old)
                    w[j] = new
                max_change = max(max_change, abs(new - old))
                max_w = max(max_w, abs(new))
            self.n_iter_ = it
            if max_w == 0.0 or max_change / max_w < self.tol:
                break             # коэффициенты перестали меняться
        return w

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
