# Utiliser une image de base Python
FROM python:3.8

# Définir le répertoire de travail
WORKDIR /app

# Copier requirements.txt first for better caching
COPY requirements.txt .

# Install packages with increased timeout and retry
#installs key large packages (numpy, pandas, and scipy) separately before installing the rest.
# This helps prevent timeouts when downloading large packages
RUN pip install --no-cache-dir --timeout=100 --retries=3 pip setuptools wheel && \
    pip install --no-cache-dir --timeout=100 --retries=3 numpy && \
    pip install --no-cache-dir --timeout=100 --retries=3 pandas && \
    pip install --no-cache-dir --timeout=100 --retries=3 scipy && \
    pip install --no-cache-dir --timeout=100 --retries=3 -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Exposer le port pour MLflow UI
EXPOSE 5000
# Expose FastAPI port
EXPOSE 8000

# Commande par défaut pour démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]