.PHONY: help data battles features train simple train-all flask streamlit tests clean

help:
	@echo "Targets:"
	@echo "  make data        build data/pokemon_stats.csv"
	@echo "  make battles     simulate 20000 battles -> data/battles.csv"
	@echo "  make features    engineer features -> data/features.csv"
	@echo "  make train       train XGBoost/LightGBM -> models/best_model.joblib"
	@echo "  make simple      train logistic regression -> models/simple_model.joblib"
	@echo "  make train-all   run data + battles + features + train + simple"
	@echo "  make flask       run the Flask app (port 5000)"
	@echo "  make streamlit   run the Streamlit app"
	@echo "  make tests       run pytest"
	@echo "  make clean       remove generated data/ and models/ artifacts"

data:
	python3 src/build_dataset.py

battles:
	python3 src/battle_simulator.py

features:
	python3 src/features.py

train: features
	python3 src/train.py

simple: features
	python3 src/simple_train.py

train-all: data battles features train simple

flask:
	python3 webapp/app.py

streamlit:
	streamlit run app.py

tests:
	pytest -q

clean:
	rm -f data/battles.csv data/features.csv
	rm -f models/*.joblib models/shap_summary.png models/calibration.png