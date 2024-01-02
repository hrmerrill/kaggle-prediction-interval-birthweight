clean:
	rm kaggle_prediction_interval_birthweight/.ipynb_checkpoints/* || true
	rm kaggle_prediction_interval_birthweight/model/.ipynb_checkpoints/* || true
	rm kaggle_prediction_interval_birthweight/workflow/.ipynb_checkpoints/* || true
	rm kaggle_prediction_interval_birthweight/data/.ipynb_checkpoints/* || true
	rm kaggle_prediction_interval_birthweight/utils/.ipynb_checkpoints/* || true
	rm kaggle_prediction_interval_birthweight/__pycache__/* || true
	rm kaggle_prediction_interval_birthweight/model/__pycache__/* || true
	rm kaggle_prediction_interval_birthweight/workflow/__pycache__/* || true
	rm kaggle_prediction_interval_birthweight/data/__pycache__/* || true
	rm kaggle_prediction_interval_birthweight/utils/__pycache__/* || true
	isort kaggle_prediction_interval_birthweight/*
	black -l 100 kaggle_prediction_interval_birthweight/*
	flake8 --max-line-length 100

build:
	poetry update
	poetry install
	poetry build
