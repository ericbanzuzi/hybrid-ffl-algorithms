include .env
export

create-env:
	@echo "⏳Creating a conda environment..."
	@python3 -m venv .venv
	@echo "Environment created!✅"

env-activate:
	@source .venv/bin/activate

init-env:
	@source .venv/bin/activate && \
	 pip install -e .
	
login:
	@echo "Logging into Weights & Biases..."
	@wandb login $(WANDB_KEY)
	@echo "Login completed!✅"

wandb-key:
	@echo $(WANDB_KEY)

fedavg-femnist-experiment:
	@SEED=$(seed) ./experiments/fedavg-femnist.sh

qfedavg-femnist-experiment:
	@SEED=$(seed) ./experiments/qfedavg-femnist.sh

fedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedavg-cifar.sh > output.log 2>&1 &

qfedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/qfedavg-cifar.sh > output.log 2>&1 &