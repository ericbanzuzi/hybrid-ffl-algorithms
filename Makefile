include .env
export

create-env:
	@echo "⏳Creating a conda environment..."
	@conda create -n venv python==3.11 && \
	 conda run -n venv pip install -r requirements.txt
	@echo "Environment created!✅"

env-activate:
	@conda activate venv

login:
	@echo "Logging into Weights & Biases..."
	@wandb login $(WANDB_KEY)
	@echo "Login completed!✅"

wandb-key:
	@echo $(WANDB_KEY)

demo-experiment:
	@echo "Starting the XX experiment..."
	@flwr run --run-config 'agg-strategy="fedprox" cli-strategy="fedprox" proximal-mu=0.5'
	@echo "Experiment completed!✅"