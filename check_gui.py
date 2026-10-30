"""Автоматическая проверка окна «РегПрогноз» без нажатий мышью.

Запуск:  python check_gui.py
Окно создаётся в невидимом режиме, скрипт сам «нажимает» кнопки и проверяет
результат и блокировки. В конце должно быть «ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ».
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # окно не показывается

# Библиотеки загружаются РАНЬШЕ PySide6 — причина описана в preload.py
import preload  # noqa: E402, F401
from core.generator import SyntheticGenerator  # noqa: E402

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402
from ui.main_window import TAB_PLOT, TAB_PREDICT, MainWindow  # noqa: E402

messages = []  # сюда попадают тексты сообщений об ошибках
QMessageBox.warning = lambda *a: messages.append(("warning", a[2]))
QMessageBox.critical = lambda *a: messages.append(("critical", a[2]))

failed = 0


def check(name, cond):
    global failed
    print(("  [ok] " if cond else "  [ОШИБКА] ") + name)
    failed += 0 if cond else 1


app = QApplication(sys.argv)
w = MainWindow()
tmp = tempfile.mkdtemp()

print("1. Начальное состояние (до загрузки данных)")
check("кнопка «Обучить» недоступна", not w.btn_train.isEnabled())
check("вкладка «Прогноз» недоступна", not w.tabs.isTabEnabled(TAB_PREDICT))
check("сохранение отчёта недоступно", not w.act_save_txt.isEnabled())

print("2. Загрузка CSV через диалог")
csv_path = os.path.join(tmp, "data.csv")
SyntheticGenerator(500, 4, 2.0, 1).generate().frame.to_csv(
    csv_path, sep=";", index=False)
QFileDialog.getOpenFileName = lambda *a, **k: (csv_path, "")
w.on_load_csv()
check("набор загружен, 500 строк", w.dataset is not None and w.dataset.row_count() == 500)
check("в таблице показано 100 строк", w.tbl_preview.rowCount() == 100)
check("цель по умолчанию — y", w.cmb_target.currentText() == "y")
check("входные признаки x1..x4", w._checked_features() == ["x1", "x2", "x3", "x4"])
check("«Обучить» стала доступна", w.btn_train.isEnabled())

print("3. Блокировки при выборе признаков")
w._check_all(False)
check("без входных признаков «Обучить» недоступна", not w.btn_train.isEnabled())
w._check_all(True)
w.cmb_target.setCurrentText("x1")
check("новая цель x1 исключена из входных", "x1" not in w._checked_features())
w.cmb_target.setCurrentText("y")
w._check_all(True)

print("4. Обучение Ridge")
w.cmb_algo.setCurrentIndex(w.cmb_algo.findData("ridge"))
check("поле α доступно для Ridge", w.sp_alpha.isEnabled())
w.on_train()
check("модель обучена", w.trained and w.result is not None)
check("таблица коэффициентов: 1 + 4 строки", w.tbl_coef.rowCount() == 5)
check("вкладки «Прогноз» и «График» открылись",
      w.tabs.isTabEnabled(TAB_PREDICT) and w.tabs.isTabEnabled(TAB_PLOT))
check("сообщений об ошибках нет", not messages)

w.cmb_algo.setCurrentIndex(w.cmb_algo.findData("linear"))
check("поле α недоступно для линейной", not w.sp_alpha.isEnabled())

print("5. Сравнение алгоритмов")
w.on_compare()
check("в таблице сравнения 3 строки", w.tbl_compare.rowCount() == 3)

print("6. Прогноз")
for edit in w.prediction_inputs.values():
    edit.setText("1,5")
w.on_predict()
check("прогноз выведен", "Прогноз" in w.lbl_prediction.text())
w.prediction_inputs["x1"].setText("abc")
w.on_predict()
check("на «abc» — сообщение об ошибке, без падения",
      messages and "не является числом" in messages[-1][1])
messages.clear()

print("7. Сохранение результатов")
for name, method in (("отчёт.txt", w.on_save_txt),
                     ("прогнозы.csv", w.on_save_predictions),
                     ("график.png", w.on_save_plot)):
    out = os.path.join(tmp, name)
    QFileDialog.getSaveFileName = lambda *a, o=out, **k: (o, "")
    method()
    check(f"создан файл {name}", os.path.isfile(out) and os.path.getsize(out) > 0)

print("8. Ошибочный файл")
bad = os.path.join(tmp, "bad.csv")
with open(bad, "w", encoding="utf-8") as f:
    f.write("a,b\n1,2\n")
QFileDialog.getOpenFileName = lambda *a, **k: (bad, "")
w.on_load_csv()
check("сообщение об ошибке показано", bool(messages))
check("прежний набор данных сохранился", w.dataset.row_count() == 500)

print("9. Генерация")
w.sp_n.setValue(300)
w.sp_p.setValue(2)
w.on_generate()
check("сгенерировано 300 строк", w.dataset.row_count() == 300)
check("коэффициенты показаны", w.txt_gen_coef.toPlainText().startswith("y ="))
check("после новых данных модель сброшена", not w.trained)

print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if failed == 0 else f"\nНЕ ПРОЙДЕНО ПРОВЕРОК: {failed}")
print(f"(файлы сохранены во временную папку {tmp})")
