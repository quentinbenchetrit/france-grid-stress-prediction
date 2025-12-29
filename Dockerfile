# Utiliser une image Python officielle légère
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /code

# Copier les requirements et installer les dépendances
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copier tout le code (votre main.py et vos fichiers modèles .pkl/.joblib s'il y en a)
COPY . /code

# Créer un utilisateur non-root (recommandé par Hugging Face pour la sécurité)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Commande de lancement
# Notez le port 7860 obligatoire pour Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]