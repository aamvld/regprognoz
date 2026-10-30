@echo off
chcp 65001 >nul
rem ===================================================================
rem  Сборка дистрибутива «РегПрогноз» (п. 4.5.4, 4.6.1 ТЗ)
rem  1) PyInstaller собирает программу в папку dist\RegPrognoz
rem     (внутри уже есть Python и все библиотеки);
rem  2) Inno Setup упаковывает её в установщик installer\RegPrognoz_Setup.exe
rem  Запуск: двойной щелчок по build.bat или команда build.bat в терминале
rem ===================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ОШИБКА] Не найдено окружение .venv. Сначала создайте его по README.md
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo === 1/3. Установка PyInstaller ===
pip install --quiet pyinstaller || goto :error

echo.
echo === 2/3. Сборка программы (1-3 минуты) ===
pyinstaller --noconfirm --clean --windowed --onedir ^
    --name RegPrognoz ^
    --add-data "examples;examples" ^
    --runtime-hook preload.py ^
    --exclude-module tkinter ^
    --exclude-module PyQt5 ^
    --exclude-module PyQt6 ^
    --exclude-module IPython ^
    --exclude-module sklearn ^
    --exclude-module scipy ^
    main.py || goto :error

echo.
echo === 3/3. Создание установщика ===
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo [!] Inno Setup 6 не найден — установщик не создан.
    echo     Программа уже собрана: dist\RegPrognoz\RegPrognoz.exe
    echo     Установите Inno Setup 6 и запустите build.bat ещё раз.
    pause
    exit /b 0
)
"%ISCC%" installer.iss || goto :error

echo.
echo ===================================================================
echo  ГОТОВО
echo  Программа:   dist\RegPrognoz\RegPrognoz.exe
echo  Установщик:  installer\RegPrognoz_Setup.exe
echo ===================================================================
pause
exit /b 0

:error
echo.
echo [ОШИБКА] Сборка прервана. Скопируйте текст выше и пришлите его.
pause
exit /b 1
