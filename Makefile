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

femnist-fedavg-experiment:
	@SEED=$(seed) ./experiments/fedavg-femnist.sh
