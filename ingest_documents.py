import os
import psycopg2
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

# Créer table documents
cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    DROP TABLE IF EXISTS documents;

    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        content TEXT,
        embedding vector(384)
    );
""")
conn.commit()

# modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # léger (~100 Mo)

# Ingestion des fichiers de documents/
document_dir = "documents"

for file_name in os.listdir(document_dir):
    file_path = os.path.join(document_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Générer l'embedding
    embedding = model.encode(content)
    print(
        f"Embedding pour {file_name} : {embedding[:10]}... (total {len(embedding)} valeurs)")

    # Insérer dans bdd psql
    cursor.execute(
        "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s)",
        (file_name, content, embedding.tolist())
    )

conn.commit()
cursor.close()
conn.close()

print("OK : Ingestion terminée")
