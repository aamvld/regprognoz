"""Проверка временных характеристик и памяти (п. 4.1.1.2 ТЗ).

Запуск:  python check_performance.py
Скрипт создаёт наборы данных максимального размера, замеряет время операций
(каждая повторяется 3 раза, в таблицу идёт ХУДШИЙ результат) и потребление
памяти, выводит таблицу «Требование | Факт» и сохраняет её в
results/performance.txt — для раздела «Тестирование» РПЗ.
"""

import io
import os
import platform
import subprocess
import sys
import tempfile
import time

import preload  # noqa: F401  (библиотеки загружаются до PySide6)
from core.csv_reader import CsvReader
from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.params import ALGORITHM_NAMES, ModelParams
from output.plot_builder import PlotBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS = 3


# ---------------------------------------------------------------------------
# Память и сведения о компьютере
# ---------------------------------------------------------------------------
def peak_memory_mb() -> float | None:
    """Пиковый объём оперативной памяти процесса, МБ."""
    try:
        import psutil  # если установлен
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        psapi = ctypes.WinDLL("psapi")
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2**20
        return None
    try:
        import resource  # Linux / macOS
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    except ImportError:
        return None


def total_ram_gb() -> str:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return f"{m.ullTotalPhys / 2**30:.1f} ГБ"
    try:
        return f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.1f} ГБ"
    except (ValueError, OSError, AttributeError):
        return "н/д"


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


# ---------------------------------------------------------------------------
# Замеры
# ---------------------------------------------------------------------------
def worst_time(action) -> float:
    """Худшее время из REPEATS запусков, секунды."""
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        action()
        times.append(time.perf_counter() - t0)
    return max(times)


def measure_startup() -> float | None:
    """Время запуска из исходников: от старта Python до готового окна.

    Окно создаётся в невидимом режиме в отдельном процессе. Время запуска
    собранного .exe замеряется вручную (секундомером) — см. итоговую таблицу.
    """
    code = (
        "import time; t0 = time.perf_counter()\n"
        "import os; os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "import sys; sys.path.insert(0, r'" + HERE + "')\n"
        "import preload\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from ui.main_window import MainWindow\n"
        "app = QApplication([]); w = MainWindow(); w.show(); app.processEvents()\n"
        "print(time.perf_counter() - t0)\n"
    )
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=HERE)
        if r.returncode != 0:
            return None
        times.append(time.perf_counter() - t0)  # включая старт интерпретатора
    return max(times)


def collect():
    """Выполняет все замеры. Возвращает (описание компьютера, строки таблицы)."""
    tmp = tempfile.mkdtemp()
    rows = []  # (операция, требование, факт, выполнено?)

    def add(name, limit, fact, unit="с"):
        """limit=None — операция ТЗ не нормируется (замер справочный)."""
        fact_text = "н/д" if fact is None else f"{fact:.2f} {unit}"
        if limit is None:
            rows.append((name, "не задано", fact_text, "—"))
        else:
            ok = fact is not None and fact <= limit
            rows.append((name, f"≤ {limit:g} {unit}", fact_text,
                         "да" if ok else ("—" if fact is None else "НЕТ")))
        print(f"  {name}: {fact_text}")

    print("Подготовка данных максимального размера…")
    # для загрузки: 10 000 строк и 10 столбцов (9 входных + целевой)
    load_path = os.path.join(tmp, "load_10000x10.csv")
    SyntheticGenerator(10_000, 9, 5.0, 1).generate().frame.to_csv(
        load_path, sep=";", index=False)
    # для обучения: 10 000 строк и 10 ВХОДНЫХ признаков
    train_ds = SyntheticGenerator(10_000, 10, 5.0, 2).generate()
    features = [f"x{i}" for i in range(1, 11)]

    print("Замеры (каждая операция ×3, берётся худшее время):")
    add("Загрузка CSV 10 000 строк × 10 столбцов", 5,
        worst_time(lambda: CsvReader().read(load_path)))

    result = None
    for algo in ("linear", "ridge", "lasso"):
        for std in (True, False):
            p = ModelParams(algorithm=algo, target="y", features=features,
                            standardize=std)

            def train(p=p):
                nonlocal result
                result = ExperimentService().train(train_ds, p)
            label = (f"Обучение: {ALGORITHM_NAMES[algo]}, 10 000 × 10, "
                     f"стандартизация {'вкл.' if std else 'выкл.'}")
            add(label, 10, worst_time(train))

    add("Сравнение трёх алгоритмов (справочно)", None, worst_time(
        lambda: ExperimentService().compare(
            train_ds, ModelParams(target="y", features=features))))

    def plot():
        fig = PlotBuilder.actual_vs_predicted(result)
        fig.savefig(io.BytesIO(), format="png")  # полная отрисовка
    add("Построение графика «факт – прогноз»", 3, worst_time(plot))

    print("Замер запуска программы (из исходников)…")
    add("Запуск программы из исходников (до готового окна)", 15, measure_startup())

    add("Пиковая память вычислительной части (этот скрипт)", 1024,
        peak_memory_mb(), "МБ")

    rows.append(("Запуск установленной программы (.exe), секундомер",
                 "≤ 15 с", "заполнить вручную", ""))
    rows.append(("Память программы (.exe) после обучения, Диспетчер задач",
                 "≤ 1024 МБ", "заполнить вручную", ""))

    env = (f"Компьютер: {cpu_name()}, ОЗУ {total_ram_gb()}, "
           f"{platform.system()} {platform.release()}, Python {platform.python_version()}")
    return env, rows


def main():
    env, rows = collect()
    w0 = max(len(r[0]) for r in rows)
    lines = [env, "",
             f"{'Операция':{w0}} | {'Требование':10} | {'Факт':17} | Выполнено",
             "-" * (w0 + 45)]
    lines += [f"{a:{w0}} | {b:10} | {c:17} | {d}" for a, b, c, d in rows]
    report = "\n".join(lines)
    print("\n" + report)

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "performance.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("Проверка временных характеристик (п. 4.1.1.2 ТЗ)\n")
        f.write(f"Дата: {time.strftime('%d.%m.%Y %H:%M')}\n\n{report}\n")
    print(f"\nТаблица сохранена: {out}")


if __name__ == "__main__":
    main()
