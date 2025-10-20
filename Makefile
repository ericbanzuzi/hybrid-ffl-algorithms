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

fedavg-femnist-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedavg-femnist.sh > output.log 2>&1 &

fedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedavg-cifar.sh > output.log 2>&1 &

qfedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/qfedavg-cifar.sh > output.log 2>&1 &