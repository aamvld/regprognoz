"""Класс MainWindow — главное окно программы «РегПрогноз».

Главное окно хранит текущие набор данных, параметры и результаты,
вызывает подсистемы ядра и блокирует недопустимые действия (п. 4.2.2 ТЗ).
"""

import os
import time
from functools import wraps

# Модули ядра (а с ними pandas) импортируются РАНЬШЕ PySide6 —
# иначе PySide6 конфликтует с библиотекой six (см. main.py).
from core.csv_reader import CsvReader
from core.dataset import Dataset, SyntheticDataset
from core.errors import ValidationError
from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.params import ALGORITHM_NAMES, ModelParams
from core.result import ModelResult
from core.validator import InputValidator
from output.plot_builder import PlotBuilder
from output.report_exporter import ReportExporter

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from ui.help_window import HelpWindow

# Номера вкладок
TAB_DATA, TAB_GEN, TAB_MODEL, TAB_COMPARE, TAB_PREDICT, TAB_PLOT = range(6)


def safe(func):
    """Декоратор для обработчиков кнопок: программа не падает (п. 4.2.3 ТЗ).

    ValidationError — ожидаемая ошибка ввода: показываем причину.
    Любая другая ошибка — непредвиденная: тоже показываем, работа продолжается.
    """
    @wraps(func)
    def wrapper(self, *args):
        try:
            return func(self)
        except ValidationError as e:
            QMessageBox.warning(self, "Ошибка ввода", str(e))
        except Exception as e:  # noqa: BLE001 — защита от аварийного завершения
            QMessageBox.critical(
                self, "Ошибка", f"Непредвиденная ошибка: {e}\n\n"
                                "Работа программы продолжается.")
        finally:
            self.update_controls_state()
    return wrapper


def fmt(v: float) -> str:
    """Число для вывода в интерфейсе."""
    return "не определён" if v != v else f"{v:.4f}"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("РегПрогноз — прогнозирование на основе регрессионного анализа")
        self.resize(1200, 720)          # помещается в экран 1366×768
        self.setMinimumSize(1000, 640)

        # --- состояние программы (атрибуты из диаграммы классов) ---
        self.dataset: Dataset | None = None
        self.generated: SyntheticDataset | None = None
        self.params = ModelParams()
        self.results: list[ModelResult] = []   # результаты сравнения
        self.result: ModelResult | None = None  # последняя обученная модель
        self.trained = False
        self.service = ExperimentService()
        self.reader = CsvReader()
        self.prediction_inputs: dict[str, QLineEdit] = {}

        self._build_menu()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Данные")
        self.tabs.addTab(self._build_gen_tab(), "Генерация")
        self.tabs.addTab(self._build_model_tab(), "Модель")
        self.tabs.addTab(self._build_compare_tab(), "Сравнение")
        self.tabs.addTab(self._build_predict_tab(), "Прогноз")
        self.tabs.addTab(self._build_plot_tab(), "График")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Загрузите файл CSV или сгенерируйте данные.")
        self.update_controls_state()

    # =====================================================================
    # Построение интерфейса
    # =====================================================================
    def _build_menu(self):
        menu_file = self.menuBar().addMenu("Файл")
        self.act_load = QAction("Загрузить CSV…", self)
        self.act_load.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load.triggered.connect(self.on_load_csv)
        self.act_save_txt = QAction("Сохранить отчёт (TXT)…", self)
        self.act_save_txt.triggered.connect(self.on_save_txt)
        self.act_save_csv = QAction("Сохранить таблицу прогнозов (CSV)…", self)
        self.act_save_csv.triggered.connect(self.on_save_predictions)
        self.act_save_png = QAction("Сохранить график (PNG)…", self)
        self.act_save_png.triggered.connect(self.on_save_plot)
        act_exit = QAction("Выход", self)
        act_exit.triggered.connect(self.close)
        for a in (self.act_load, None, self.act_save_txt, self.act_save_csv,
                  self.act_save_png, None, act_exit):
            menu_file.addSeparator() if a is None else menu_file.addAction(a)

        menu_help = self.menuBar().addMenu("Справка")
        act_help = QAction("Справка", self)
        act_help.setShortcut(QKeySequence("F1"))
        act_help.triggered.connect(self.on_help)
        menu_help.addAction(act_help)

    # ---------- вкладка «Данные» ----------
    def _build_data_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        btn_load = QPushButton("Загрузить CSV…")
        btn_load.clicked.connect(self.on_load_csv)
        btn_gen = QPushButton("Сгенерировать данные…")
        btn_gen.clicked.connect(lambda: self.tabs.setCurrentIndex(TAB_GEN))
        top.addWidget(btn_load)
        top.addWidget(btn_gen)
        top.addStretch()
        lay.addLayout(top)

        self.lbl_data_info = QLabel("Набор данных не загружен.")
        lay.addWidget(self.lbl_data_info)

        self.tbl_preview = QTableWidget()
        self.tbl_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tbl_preview)
        return w

    # ---------- вкладка «Генерация» ----------
    def _build_gen_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Параметры генерации (п. 4.1.2.2 ТЗ)")
        form = QFormLayout(box)

        self.sp_n = QSpinBox()
        self.sp_n.setRange(20, 10_000)
        self.sp_n.setValue(500)
        self.sp_p = QSpinBox()
        self.sp_p.setRange(1, 20)
        self.sp_p.setValue(3)
        self.sp_noise = QDoubleSpinBox()
        self.sp_noise.setRange(0.0, 1_000.0)
        self.sp_noise.setDecimals(2)
        self.sp_noise.setValue(1.0)
        self.sp_gen_seed = QSpinBox()
        self.sp_gen_seed.setRange(0, 2_147_483_647)
        self.sp_gen_seed.setValue(42)
        form.addRow("Число наблюдений:", self.sp_n)
        form.addRow("Число входных признаков:", self.sp_p)
        form.addRow("Уровень шума σ:", self.sp_noise)
        form.addRow("Начальное значение генератора:", self.sp_gen_seed)
        lay.addWidget(box)

        row = QHBoxLayout()
        btn = QPushButton("Сгенерировать")
        btn.clicked.connect(self.on_generate)
        self.btn_save_gen = QPushButton("Сохранить данные в CSV…")
        self.btn_save_gen.clicked.connect(self.on_save_generated)
        row.addWidget(btn)
        row.addWidget(self.btn_save_gen)
        row.addStretch()
        lay.addLayout(row)

        lay.addWidget(QLabel("Истинная зависимость (коэффициенты сгенерированы программой):"))
        self.txt_gen_coef = QTextEdit()
        self.txt_gen_coef.setReadOnly(True)
        self.txt_gen_coef.setMaximumHeight(120)
        lay.addWidget(self.txt_gen_coef)
        lay.addStretch()
        return w

    # ---------- вкладка «Модель» ----------
    def _build_model_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        # левая колонка: признаки и параметры
        left = QVBoxLayout()
        box_f = QGroupBox("Признаки")
        lf = QVBoxLayout(box_f)
        lf.addWidget(QLabel("Целевой признак:"))
        self.cmb_target = QComboBox()
        self.cmb_target.currentTextChanged.connect(self.on_target_changed)
        lf.addWidget(self.cmb_target)
        lf.addWidget(QLabel("Входные признаки:"))
        self.lst_features = QListWidget()
        self.lst_features.itemChanged.connect(lambda _: self.update_controls_state())
        lf.addWidget(self.lst_features)
        row = QHBoxLayout()
        b_all = QPushButton("Выбрать все")
        b_all.clicked.connect(lambda: self._check_all(True))
        b_none = QPushButton("Снять все")
        b_none.clicked.connect(lambda: self._check_all(False))
        row.addWidget(b_all)
        row.addWidget(b_none)
        lf.addLayout(row)
        self.lbl_missing = QLabel("")
        lf.addWidget(self.lbl_missing)
        left.addWidget(box_f)

        box_p = QGroupBox("Параметры модели (п. 4.1.2.3 ТЗ)")
        form = QFormLayout(box_p)
        self.cmb_algo = QComboBox()
        for key, name in ALGORITHM_NAMES.items():
            self.cmb_algo.addItem(name, key)
        self.cmb_algo.currentIndexChanged.connect(lambda _: self.update_controls_state())
        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setDecimals(4)
        self.sp_alpha.setRange(0.0001, 100_000.0)
        self.sp_alpha.setValue(1.0)
        self.chk_std = QCheckBox("включена")
        self.chk_std.setChecked(True)
        self.sp_test = QSpinBox()
        self.sp_test.setRange(10, 40)
        self.sp_test.setValue(20)
        self.sp_test.setSuffix(" %")
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_147_483_647)
        self.sp_seed.setValue(42)
        self.cmb_missing = QComboBox()
        self.cmb_missing.addItem("удаление строк с пропусками", "drop")
        self.cmb_missing.addItem("замена средним значением", "mean")
        form.addRow("Алгоритм:", self.cmb_algo)
        form.addRow("Параметр α:", self.sp_alpha)
        form.addRow("Стандартизация:", self.chk_std)
        form.addRow("Доля тестовой выборки:", self.sp_test)
        form.addRow("Начальное значение:", self.sp_seed)
        form.addRow("Пропуски:", self.cmb_missing)
        left.addWidget(box_p)

        row2 = QHBoxLayout()
        self.btn_train = QPushButton("Обучить модель")
        self.btn_train.clicked.connect(self.on_train)
        self.btn_compare = QPushButton("Сравнить алгоритмы")
        self.btn_compare.clicked.connect(self.on_compare)
        row2.addWidget(self.btn_train)
        row2.addWidget(self.btn_compare)
        left.addLayout(row2)

        # правая колонка: результаты
        right = QVBoxLayout()
        box_r = QGroupBox("Результаты обучения")
        lr = QVBoxLayout(box_r)
        self.lbl_equation = QLabel("Модель не обучена.")
        self.lbl_equation.setWordWrap(True)
        self.lbl_equation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lr.addWidget(self.lbl_equation)
        self.tbl_coef = QTableWidget(0, 2)
        self.tbl_coef.setHorizontalHeaderLabels(["Признак", "Коэффициент"])
        self.tbl_coef.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_coef.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lr.addWidget(self.tbl_coef)
        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setStyleSheet("font-size: 14px;")
        lr.addWidget(self.lbl_metrics)
        right.addWidget(box_r)

        lay.addLayout(left, 2)
        lay.addLayout(right, 3)
        return w

    # ---------- вкладка «Сравнение» ----------
    def _build_compare_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Сравнение алгоритмов на одном наборе данных "
                             "(одинаковое разбиение и параметры). "
                             "Лучшие значения выделены."))
        self.tbl_compare = QTableWidget(0, 4)
        self.tbl_compare.setHorizontalHeaderLabels(["Алгоритм", "MSE", "MAE", "R²"])
        self.tbl_compare.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_compare.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tbl_compare)
        return w

    # ---------- вкладка «Прогноз» ----------
    def _build_predict_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Введите значения всех входных признаков обученной модели:"))
        self.predict_form_box = QGroupBox("Наблюдение")
        self.predict_form = QFormLayout(self.predict_form_box)
        lay.addWidget(self.predict_form_box)
        self.btn_predict = QPushButton("Вычислить прогноз")
        self.btn_predict.clicked.connect(self.on_predict)
        lay.addWidget(self.btn_predict)
        self.lbl_prediction = QLabel("")
        self.lbl_prediction.setStyleSheet("font-size: 16px; font-weight: bold;")
        lay.addWidget(self.lbl_prediction)
        lay.addStretch()
        return w

    # ---------- вкладка «График» ----------
    def _build_plot_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay.addWidget(self.canvas)
        self.btn_save_png = QPushButton("Сохранить график (PNG)…")
        self.btn_save_png.clicked.connect(self.on_save_plot)
        lay.addWidget(self.btn_save_png)
        return w

    # =====================================================================
    # Блокировка недопустимых действий (п. 4.2.2 ТЗ)
    # =====================================================================
    def update_controls_state(self):
        has_data = self.dataset is not None
        target = self.cmb_target.currentText()
        features = self._checked_features()
        can_train = has_data and bool(target) and bool(features)

        self.btn_train.setEnabled(can_train)
        self.btn_compare.setEnabled(can_train)

        algo = self.cmb_algo.currentData()
        self.sp_alpha.setEnabled(algo in ("ridge", "lasso"))

        # способ обработки пропусков нужен, только если они есть
        if has_data and target:
            n_miss = self.dataset.missing_count([target] + features)
            self.cmb_missing.setEnabled(n_miss > 0)
            self.lbl_missing.setText(
                f"Пропусков в выбранных столбцах: {n_miss}" if n_miss else "")
        else:
            self.cmb_missing.setEnabled(False)

        # прогноз, график и сохранение — только после обучения
        self.tabs.setTabEnabled(TAB_PREDICT, self.trained)
        self.tabs.setTabEnabled(TAB_PLOT, self.trained)
        for a in (self.act_save_txt, self.act_save_csv, self.act_save_png):
            a.setEnabled(self.trained)
        self.btn_save_gen.setEnabled(self.generated is not None)

    # =====================================================================
    # Работа с данными
    # =====================================================================
    @safe
    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить набор данных", "", "Файлы CSV (*.csv);;Все файлы (*)")
        if not path:
            return  # пользователь нажал «Отмена»
        t0 = time.perf_counter()
        ds = self.reader.read(path)
        self._set_dataset(ds)
        self.statusBar().showMessage(
            f"Файл загружен за {time.perf_counter() - t0:.2f} с: {path}")

    @safe
    def on_generate(self):
        gen = SyntheticGenerator(self.sp_n.value(), self.sp_p.value(),
                                 self.sp_noise.value(), self.sp_gen_seed.value())
        ds = gen.generate()
        self.generated = ds
        self.txt_gen_coef.setPlainText(ds.coefficients_text())
        self._set_dataset(ds)
        self.statusBar().showMessage(
            f"Сгенерировано {ds.row_count()} наблюдений, "
            f"{self.sp_p.value()} признаков.")

    @safe
    def on_save_generated(self):
        path = self._ask_save_path("Сохранить сгенерированные данные",
                                   "synthetic.csv", "Файлы CSV (*.csv)")
        if path:
            SyntheticGenerator.save_csv(self.generated, path)
            self.statusBar().showMessage(f"Данные сохранены: {path}")

    def _set_dataset(self, ds: Dataset):
        """Делает набор текущим и обновляет все зависящие от него элементы."""
        self.dataset = ds
        self.trained = False          # старая модель к новым данным не относится
        self.result = None
        self.results = []
        self.tbl_compare.setRowCount(0)
        self.lbl_equation.setText("Модель не обучена.")
        self.tbl_coef.setRowCount(0)
        self.lbl_metrics.setText("")

        # информация и предпросмотр 100 строк
        src = os.path.basename(ds.source_path) if ds.source_path else "синтетические данные"
        self.lbl_data_info.setText(
            f"Источник: {src}.   Строк: {ds.row_count()},  "
            f"столбцов: {ds.column_count()},  пропусков: {ds.missing_count()}.  "
            f"Показаны первые {min(100, ds.row_count())} строк.")
        prev = ds.preview(100)
        self.tbl_preview.clear()
        self.tbl_preview.setRowCount(len(prev))
        self.tbl_preview.setColumnCount(prev.shape[1])
        self.tbl_preview.setHorizontalHeaderLabels([str(c) for c in prev.columns])
        for i, row in enumerate(prev.itertuples(index=False)):
            for j, v in enumerate(row):
                text = "—" if v != v else str(v)   # NaN показываем прочерком
                self.tbl_preview.setItem(i, j, QTableWidgetItem(text))

        # списки признаков: по умолчанию цель — последний числовой столбец
        numeric = ds.numeric_columns()
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        self.cmb_target.addItems(numeric)
        self.cmb_target.setCurrentIndex(len(numeric) - 1)
        self.cmb_target.blockSignals(False)
        self.lst_features.blockSignals(True)
        self.lst_features.clear()
        for name in numeric:
            item = QListWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked)
            self.lst_features.addItem(item)
        self.lst_features.blockSignals(False)
        self.on_target_changed(self.cmb_target.currentText())
        self.tabs.setCurrentIndex(TAB_DATA)

    def on_target_changed(self, target: str):
        """Целевой признак нельзя выбрать входным: его пункт блокируется."""
        self.lst_features.blockSignals(True)
        for i in range(self.lst_features.count()):
            item = self.lst_features.item(i)
            if item.text() == target:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable)          # недоступен
            else:
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        self.lst_features.blockSignals(False)
        self.update_controls_state()

    def _checked_features(self) -> list[str]:
        target = self.cmb_target.currentText()
        return [self.lst_features.item(i).text()
                for i in range(self.lst_features.count())
                if self.lst_features.item(i).checkState() == Qt.CheckState.Checked
                and self.lst_features.item(i).text() != target]

    def _check_all(self, state: bool):
        target = self.cmb_target.currentText()
        self.lst_features.blockSignals(True)
        for i in range(self.lst_features.count()):
            item = self.lst_features.item(i)
            if item.text() != target:
                item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
        self.lst_features.blockSignals(False)
        self.update_controls_state()

    # =====================================================================
    # Моделирование
    # =====================================================================
    def _read_params(self) -> ModelParams:
        """Собирает параметры из элементов интерфейса."""
        return ModelParams(
            algorithm=self.cmb_algo.currentData(),
            target=self.cmb_target.currentText(),
            features=self._checked_features(),
            alpha=self.sp_alpha.value(),
            standardize=self.chk_std.isChecked(),
            test_size=self.sp_test.value() / 100,
            seed=self.sp_seed.value(),
            missing_method=self.cmb_missing.currentData(),
        )

    @safe
    def on_train(self):
        self.params = self._read_params()
        t0 = time.perf_counter()
        self.result = self.service.train(self.dataset, self.params)
        dt = time.perf_counter() - t0
        self.trained = True
        self._show_result(self.result)
        self._build_prediction_form(self.params.features)
        self._draw_plot()
        self.statusBar().showMessage(f"Модель обучена за {dt:.2f} с.")

    @safe
    def on_compare(self):
        p = self._read_params()
        t0 = time.perf_counter()
        self.results = self.service.compare(self.dataset, p)
        rows = [r.summary_row() for r in self.results]
        self.tbl_compare.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.tbl_compare.setItem(i, 0, QTableWidgetItem(row[0]))
            for j in (1, 2, 3):
                self.tbl_compare.setItem(i, j, QTableWidgetItem(fmt(row[j])))
        # подсветка лучших: минимальные MSE и MAE, максимальный R²
        for j, best in ((1, min), (2, min), (3, max)):
            vals = [row[j] for row in rows if row[j] == row[j]]
            if vals:
                b = best(vals)
                for i, row in enumerate(rows):
                    if row[j] == b:
                        # светло-зелёный фон + тёмный жирный текст: читается
                        # и в светлой, и в тёмной теме Windows
                        cell = self.tbl_compare.item(i, j)
                        cell.setBackground(QColor("#b9e6b9"))
                        cell.setForeground(QColor("#0b3d0b"))
                        font = cell.font()
                        font.setBold(True)
                        cell.setFont(font)
        self.tabs.setCurrentIndex(TAB_COMPARE)
        self.statusBar().showMessage(
            f"Сравнение выполнено за {time.perf_counter() - t0:.2f} с.")

    def _show_result(self, r: ModelResult):
        self.lbl_equation.setText(
            f"<b>{ALGORITHM_NAMES[r.params.algorithm]}</b><br>"
            f"Уравнение: {r.equation_text()}")
        names = ["Свободный член"] + r.params.features
        values = [r.intercept] + r.coefficients
        self.tbl_coef.setRowCount(len(names))
        for i, (n, v) in enumerate(zip(names, values)):
            self.tbl_coef.setItem(i, 0, QTableWidgetItem(n))
            self.tbl_coef.setItem(i, 1, QTableWidgetItem(f"{v:.6f}"))
        m = r.metrics
        self.lbl_metrics.setText(
            f"Показатели на тестовой выборке ({len(r.y_test)} наблюдений):<br>"
            f"<b>MSE</b> = {fmt(m.mse)} &nbsp;&nbsp; <b>MAE</b> = {fmt(m.mae)}"
            f" &nbsp;&nbsp; <b>R²</b> = {fmt(m.r2)}")

    # =====================================================================
    # Прогноз и график
    # =====================================================================
    def _build_prediction_form(self, features: list[str]):
        """Поля ввода по одному на каждый входной признак модели."""
        while self.predict_form.rowCount():
            self.predict_form.removeRow(0)
        self.prediction_inputs = {}
        for f in features:
            edit = QLineEdit()
            edit.setPlaceholderText("число, например 1.5")
            self.predict_form.addRow(f"{f}:", edit)
            self.prediction_inputs[f] = edit
        self.lbl_prediction.setText("")

    @safe
    def on_predict(self):
        raw = {f: e.text() for f, e in self.prediction_inputs.items()}
        values = InputValidator.validate_prediction_input(
            raw, self.result.params.features)
        y_hat = self.service.predict_one(values)
        self.lbl_prediction.setText(
            f"Прогноз «{self.result.params.target}» = {y_hat:.4f}")

    def _draw_plot(self):
        t0 = time.perf_counter()
        PlotBuilder.actual_vs_predicted(self.result, self.figure)
        self.canvas.draw()
        return time.perf_counter() - t0

    # =====================================================================
    # Экспорт и справка
    # =====================================================================
    def _ask_save_path(self, title: str, default: str, filt: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        return path

    @safe
    def on_save_txt(self):
        path = self._ask_save_path("Сохранить отчёт", "отчёт.txt", "Текст (*.txt)")
        if path:
            src = self.dataset.source_path or ""
            ReportExporter.save_txt(self.result, path, src, self.dataset.row_count())
            self.statusBar().showMessage(f"Отчёт сохранён: {path}")

    @safe
    def on_save_predictions(self):
        path = self._ask_save_path("Сохранить таблицу прогнозов",
                                   "прогнозы.csv", "Файлы CSV (*.csv)")
        if path:
            ReportExporter.save_predictions_csv(self.result, path)
            self.statusBar().showMessage(f"Таблица сохранена: {path}")

    @safe
    def on_save_plot(self):
        path = self._ask_save_path("Сохранить график", "график.png", "PNG (*.png)")
        if path:
            ReportExporter.save_plot_png(self.figure, path)
            self.statusBar().showMessage(f"График сохранён: {path}")

    def on_help(self, *args):
        HelpWindow(self).exec()
