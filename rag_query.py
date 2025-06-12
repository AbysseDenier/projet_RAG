import psycopg2
import subprocess
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Connexion à PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_db",
    user="rag_user",
    password="rag_password"
)
cursor = conn.cursor()

# Charger le modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Demander un prompt à l’utilisateur
prompt = input("Pose ta question : ")

# Embedding du prompt
prompt_embedding = model.encode(prompt).tolist()

# Requête SQL pgvector : top 3 documents les plus proches (cosine distance)
cursor.execute(
    "SELECT content FROM documents ORDER BY embedding <=> %s::vector LIMIT 3;",
    (prompt_embedding,)
)

# Récupérer les documents les plus proches
results = cursor.fetchall()
context = "\n\n".join([row[0] for row in results])

# Construction du prompt enrichi pour le LLM
enriched_prompt = f"""
Tu es un assistant intelligent.
Voici des documents de référence : 

{context}

Question : {prompt}
Réponds de manière concise et précise.
"""

# Appel à Ollama via le CLI dans le container Docker
try:
    process = subprocess.Popen(
        ["docker", "exec", "-i", "ollama", "ollama", "run", "gemma:2b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, error = process.communicate(enriched_prompt)

    if process.returncode != 0:
        answer = f"Erreur de génération LLM : {error.strip()}"
    else:
        answer = output.strip()
except Exception as e:
    answer = f"Erreur lors de l’appel à Ollama via Docker : {e}"

# Affichage de la réponse
print("\n Réponse du LLM :")
print(answer)

# Nettoyage
cursor.close()
conn.close()
