# Utiliser une image de base Python
FROM python:3.8

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour MLflow UI
EXPOSE 5000

# Commande par défaut pour démarrer MLflow Tracking Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


