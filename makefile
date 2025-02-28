# Variables
TRAINING_SCRIPT = main.py
MODEL_FILE = churn_model.joblib
VENV_DIR = venv/
MLFLOW_PORT = 5000
MLFLOW_BACKEND_STORE = sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT = ./mlruns

# Define variables for the image name and tag
IMAGE_NAME=asinet/mlflow-server
TAG=latest


# Step 1: Install dependencies and create virtual environment
install:
	@echo "🔄 Installing dependencies and setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip
	. $(VENV_DIR)/bin/activate && pip install -r requirements.txt
	@echo "✅ Virtual environment setup complete. Please run: 'source $(VENV_DIR)/bin/activate' to activate the virtual environment."

# Step 2: Check code quality (formatting, linting, security)
lint:
	@echo "🔍 Checking code quality (formatting, linting, security)..."
	. $(VENV_DIR)/bin/activate && black .
	. $(VENV_DIR)/bin/activate && flake8 .
	. $(VENV_DIR)/bin/activate && bandit -r .
	@echo "✅ Code quality checks complete."

# Step 3: Prepare the data
prepare:
	@echo "🔄 Preparing data..."
	. $(VENV_DIR)/bin/activate && python $(TRAINING_SCRIPT) --prepare
	@echo "✅ Data preparation complete."

# Step 4: Train the model
train:
	@echo "🔄 Training the model..."
	. $(VENV_DIR)/bin/activate && python $(TRAINING_SCRIPT) --train
	@echo "✅ Model training complete."

# Step 5: Run basic tests
test:
	@echo "🔄 Running tests..."
	. $(VENV_DIR)/bin/activate && pytest tests/
	@echo "✅ Tests complete."

# Step 6: Evaluate the trained model
evaluate:
	@echo "🔄 Evaluating the model..."
	. $(VENV_DIR)/bin/activate && python $(TRAINING_SCRIPT) --evaluate
	@echo "✅ Model evaluation complete."

# Step 7: Save the model
save:
	@echo "🔄 Saving the model..."
	. $(VENV_DIR)/bin/activate && python $(TRAINING_SCRIPT) --save $(MODEL_FILE)
	@echo "✅ Model saved as $(MODEL_FILE)."

# Step 8: Run the API for testing (open Swagger UI)

api:
	@echo "🔄 Starting the API server..."
	. $(VENV_DIR)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Step 8: Run the API for testing (open Swagger UI)
#api:
#	@echo "🔄 Starting the API server..."
#	. $(VENV_DIR)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
#	@sleep 2
#	@echo "🔍 Opening Swagger UI..."
#	@xdg-open http://0.0.0.0:8000/docs

 #Démarrer le serveur MLflow UI
mlflow_ui:
	@echo "🔄 Démarrage du serveur MLflow UI..."
	mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_STORE) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT)
	@echo "✅ Serveur MLflow UI démarré sur http://localhost:$(MLFLOW_PORT)"


#Automate with a Makefile


# Build the Docker image from the Dockerfile
# The '-t' flag assigns a name and tag to the image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Tag the image to ensure it follows the correct naming convention
# This step is useful when renaming the image before pushing
tag:
	docker tag $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):$(TAG)

# Push the image to Docker Hub
# This makes the image available for download on other machines
push:
	docker push $(IMAGE_NAME):$(TAG)

# Run a new container from the image
# '-d' runs the container in detached mode (in the background)
# '--name mlflow-server' assigns a name to the running container
# '-p 5000:5000' maps port 5000 of the container to port 5000 on the host machine
run:
	docker run -d --name mlflow-server -p 5000:5000 $(IMAGE_NAME):$(TAG)

# Clean up Docker resources
# '-f' forces removal to avoid errors if the container/image doesn't exist
clean:
	# Stop and remove the running container if it exists
	docker rm -f mlflow-server || true

	# Remove the Docker image to free up space
	docker rmi -f $(IMAGE_NAME):$(TAG) || true