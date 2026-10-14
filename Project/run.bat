@echo off
setlocal
python -m pip install -r requirements.txt
python train_model.py
python evaluate.py
python eda.py
python app.py
