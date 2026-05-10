.PHONY: install dev test lint run build-graph

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check src tests app.py scripts

run:
	streamlit run app.py

build-graph:
	python scripts/build_graph.py --place "Barão Geraldo, Campinas, Brazil" --network drive --out data/graph.graphml
