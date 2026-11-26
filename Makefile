# Fraud Guard — common commands (Windows: use `make` via chocolatey/scoop, or run recipes via py)

.PHONY: setup data eda preprocess train baselines evaluate api frontend test docker-up docker-down clean

setup:
	py -m pip install -r requirements.txt

data:
	py -m src.data.download

eda:
	py -m src.data.eda

preprocess:
	py -m src.data.preprocess

split: preprocess

train:
	py -m src.train_web_bundle
	py -m src.train

baselines:
	py -m src.models.baselines

evaluate:
	py -m src.evaluate

api:
	py -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

test:
	py -m pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	py -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
