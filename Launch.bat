@echo off

REM Activation de l’environnement conda via Miniforge
call "C:\ProgramData\miniforge3\Scripts\conda.bat" activate Psychopy_template

REM Aller dans le dossier du projet
cd /d "E:\protocols\Psychopy_template"

REM Exécuter le script Python
python main.py

pause
