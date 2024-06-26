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
	poetry run isort kaggle_prediction_interval_birthweight/*
	poetry run black -l 100 kaggle_prediction_interval_birthweight/*
	poetry run flake8 --max-line-length 100

build:
	poetry update
	poetry install
	poetry build

firedrill:
	make clean || true
	make build || true
	git add .
	git commit -m "saving work"
	git push origin main

run:
	poetry run kaggle-prediction-interval-birthweight create-submission RidgeRegressor
	poetry run kaggle-prediction-interval-birthweight create-submission HistBoostRegressor
	poetry run kaggle-prediction-interval-birthweight create-submission WildWoodRegressor
	poetry run kaggle-prediction-interval-birthweight create-submission MissingnessNeuralNetRegressor
	poetry run kaggle-prediction-interval-birthweight create-submission MissingnessNeuralNetClassifier
	poetry run kaggle-prediction-interval-birthweight create-submission MissingnessNeuralNetEIM
	poetry run kaggle-prediction-interval-birthweight create-submission HistBoostEnsembler
	poetry run kaggle-prediction-interval-birthweight create-submission NeuralNetEnsembler
	poetry run kaggle-prediction-interval-birthweight create-hail-mary-submission