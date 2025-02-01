# Set the environment name
ENV_NAME=my_env
PYTHON_VERSION=3.10.16  # Change as needed

# Create a new Conda environment
create_env:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y

# Install Rust (if missing)
install_rust:
	command -v rustc >/dev/null 2>&1 || sudo curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source $$HOME/.cargo/env

# Install dependencies from requirements.txt
install_reqs:
	python -m pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt
	python -m ipykernel install --user --name=$(ENV_NAME) --display-name "Python ($(ENV_NAME))"

# Export installed dependencies to requirements.txt
export_reqs:
	conda activate $(ENV_NAME) && pip freeze > requirements.txt

# Export Conda environment to an environment.yml file
export_env:
	conda env export --name $(ENV_NAME) --from-history > environment.yml

# Sync dependencies between Conda and requirements.txt
sync:
	make export_reqs && make export_env

# Remove the Conda environment
clean:
	conda remove --name $(ENV_NAME) --all -y