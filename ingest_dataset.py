import psycopg2
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Connexion à PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_db",
    user="rag_user",
    password="rag_password"
)
cursor = conn.cursor()

# Créer la table dataset
cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    DROP TABLE IF EXISTS dataset;

    CREATE TABLE dataset (
        id SERIAL PRIMARY KEY,
        question TEXT,
        context TEXT,
        answer TEXT,
        embedding vector(384)
    );
""")
conn.commit()

# Charger le dataset HuggingFace
print("Chargement du dataset...")
dataset = load_dataset("neural-bridge/rag-dataset-12000",
                       split="test").select(range(500))     # 500 exemples à ingérer

# Modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ingestion des exemples
print(f"Ingestion de {len(dataset)} exemples dans la table 'dataset'...")
for i, sample in enumerate(dataset):
    question = sample["question"]
    context = sample["context"]
    answer = sample["answer"]

    embedding = model.encode(context)

    cursor.execute(
        "INSERT INTO dataset (question, context, answer, embedding) VALUES (%s, %s, %s, %s)",
        (question, context, answer, embedding.tolist())
    )

    if i % 100 == 0:
        print(f"{i} exemples insérés...")

conn.commit()
cursor.close()
conn.close()
print("OK : Ingestion terminée dans la table 'dataset'")
