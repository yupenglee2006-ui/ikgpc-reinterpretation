.PHONY: install test reproduce figures clean

install:
	python -m pip install -e ".[test]"

test:
	python -m pytest -q

reproduce:
	python scripts/run_all.py

figures:
	python scripts/01_synthetic_1d.py
	python scripts/02_jura_estimation.py
	python scripts/06_figure4.py

clean:
	rm -rf results/figures/* results/tables/* results/*.npz results/run_record.json
