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

fedprox-femnist-experiment:
	@SEED=$(seed) ./experiments/fedprox-femnist.sh

qfedavg-femnist-experiment:
	@SEED=$(seed) ./experiments/qfedavg-femnist.sh

fedyogi-femnist-experiment:
	@SEED=$(seed) ./experiments/fedyogi-femnist.sh

ditto-femnist-experiment:
	@SEED=$(seed) ./experiments/ditto-femnist.sh

fedproxyogi-femnist-experiment:
	@SEED=$(seed) ./experiments/fedproxyogi-femnist.sh

fedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedavg-cifar.sh > output.log 2>&1 &

fedprox-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedprox-cifar.sh > output2.log 2>&1 &

qfedavg-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/qfedavg-cifar.sh > output.log 2>&1 &

fedyogi-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedyogi-cifar.sh > output.log 2>&1 &

ditto-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/ditto-cifar.sh > output.log 2>&1 &

fedproxyogi-cifar-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedproxyogi-cifar.sh > output.log 2>&1 &

fedavg-text-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedavg-text.sh > output.log 2>&1 &

fedprox-text-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedprox-text.sh > output.log 2>&1 &

qfedavg-text-experiment:
	@nohup env SEED=$(seed) bash ./experiments/qfedavg-text.sh > output.log 2>&1 &

fedyogi-text-experiment:
	@nohup env SEED=$(seed) bash ./experiments/fedyogi-text.sh > output.log 2>&1 &

ditto-text-experiment:
	@nohup env SEED=$(seed) bash ./experiments/ditto-text.sh > output.log 2>&1 &

fedproxyogi-text-experiment:
	@nohup env SEED=$(seed) ./experiments/fedproxyogi-text.sh > output.log 2>&1 &

fedyogiditto-text-experiment:
	@nohup env SEED=$(seed) ./experiments/fedyogiditto-text.sh > output.log 2>&1 &