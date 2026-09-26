@echo off
chcp 65001 >nul
rem Windows: installs what gdbot needs, then runs a quick self-test in the simulator.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Не найден Python. Установите Python 3.10 или новее с https://www.python.org/downloads/
  echo и в установщике отметьте галочку "Add python.exe to PATH". Потом запустите этот файл снова.
  pause
  exit /b 1
)

echo === Устанавливаю зависимости ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Установка не удалась - текст ошибки выше.
  pause
  exit /b 1
)

echo.
echo === Самопроверка: бот учится проходить тестовый уровень в симуляторе ===
python -m gdbot solve -c configs\sim.json
echo.
echo Если выше написано CLEARED - всё работает.
pause
