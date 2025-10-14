@echo off
echo Starting Moffitt Researcher Assistant...
set PYTHONPATH=%PYTHONPATH%;%CD%\src
streamlit run app.py