# RAG LLM Project – Projet IA local avec PostgreSQL, pgvector, Ollama et Gemma

## Objectifs

Mettre en place un système **RAG (Retrieval-Augmented Generation)** entièrement **local**, combinant :
- un **LLM local** (`gemma:2b` via Ollama),
- une **base vectorielle PostgreSQL + pgvector**,
- des scripts Python pour indexer et interroger des documents ou des jeux de données,
- une **évaluation automatique** des performances avec métriques.

## Technologies utilisées

- Docker & Docker Compose
- PostgreSQL + pgvector
- Ollama (`gemma:2b`)
- Python 3
- Sentence Transformers (`all-MiniLM-L6-v2`)
- `psycopg2`, `requests`, `datasets`, `csv`

## Installation et exécution du projet

### 1. Prérequis

- Docker et Docker Compose installés  
- Python3 installé

### 2. Cloner le repo

```bash
git clone https://github.com/AbysseDenier/projet_RAG.git
cd projet_RAG
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

### 9. Indexation d’un dataset HuggingFace (rag-dataset-12000)
Ingestion des exemples :
```bash
python ingest_dataset.py
```
Ce script :
- télécharge rag-dataset-12000 via HuggingFace
- insère 100 exemples dans la table dataset avec vecteurs
- supprime les anciennes données de la table dataset

### 10. Évaluation automatique du RAG sur le dataset
```bash
python rag_query_and_evaluate_dataset.py
```
Ce script :
- sélectionne 5 questions de la table dataset aléatoirement
- interroge le LLM via RAG
- compare la réponse générée à la réponse attendue
- calcule une similarité cosinus
- exporte les résultats dans evaluation_rag_dataset.csv

Exemple de sortie dans le terminal :
```
(1/5) Question : What is the Berry Export Summary 2028 and its purpose?
Réponse attendue : The Berry Export Summary 2028 is a dedicated export plan...
Réponse RAG : The Berry Export Summary 2028 is a roadmap to grow exports...
Similarité cosinus : 0.82
```
