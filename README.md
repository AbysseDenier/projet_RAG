# RAG LLM Project – Projet IA local avec PostgreSQL, pgvector, Ollama et Gemma

## Objectifs

Ce projet met en œuvre un système de **RAG** (Retrieval-Augmented Generation) **100% local**, utilisant :
- un **LLM local** (Gemma via Ollama),
- une **base de connaissances vectorielle** (PostgreSQL + pgvector),
- et des scripts Python pour interroger le LLM avec enrichissement contextuel.

## Technologies utilisées

- Docker & Docker Compose
- PostgreSQL + extension `pgvector`
- Ollama (`gemma:2b`)
- Python 3.11
- Sentence Transformers (`all-MiniLM-L6-v2`)
- `psycopg2`, `requests`

## Installation et exécution du projet

### 1. Prérequis

- Docker et Docker Compose installés  
- Python3 installé

### 2. Cloner le repo

```bash
git clone https://github.com/ton_utilisateur/ton_repo_rag.git
cd ton_repo_rag
```

### 3. Créer et activer un environnement virtuel Python

```bash
python3.11 -m venv venv_rag_project
source venv_rag_project/bin/activate
```

### 4. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 5. Lancer l’infrastructure via Docker

Démarrer **PostgreSQL avec pgvector** et **Ollama** via `docker-compose.yml`.

```bash
docker-compose up -d
```

Le fichier `docker-compose.yml` instancie :
- le conteneur `ollama` exposé sur le port `11434`
- le conteneur `pgvector` avec une base `rag_db` sur le port `5433`

### 6. Ajouter les fichiers à indexer

Placer vos fichiers `.txt` dans le dossier `documents/`.  
Ils seront vectorisés et ajoutés à la base de données.

### 7. Ingestion des documents

Exécuter le script d’ingestion :

```bash
python ingest_documents.py
```

Ce script :
- vectorise chaque document avec `all-MiniLM-L6-v2`
- insère les embeddings dans `documents`

### 8. Interroger le système RAG

Lancer le script de requête :

```bash
python rag_query.py
```

Ce script :
- demande une question à l'utilisateur
- génère l’embedding de la question
- récupère les 3 documents les plus proches dans PostgreSQL
- génère un **prompt enrichi** et l’envoie à Ollama (Gemma)
- affiche la réponse du LLM

Exemple :
```
Pose ta question : Où est Paris ?

Réponse du LLM :
Paris est la capitale de la France. 
Ce document est donné comme référence.
```

## Extensions possibles

- Ajouter LangChain pour orchestrer les étapes RAG
- Indexer un dataset HuggingFace (ex. : rag-dataset-12000)
- Ajouter une interface web simple (Streamlit, Flask)
