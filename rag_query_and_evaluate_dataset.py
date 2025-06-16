import psycopg2
import subprocess
from sentence_transformers import SentenceTransformer, util
import os
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Connexion PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_db",
    user="rag_user",
    password="rag_password"
)
cursor = conn.cursor()

# Charger le modèle SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sélectionner 5 exemples aléatoires depuis la table dataset
cursor.execute(
    "SELECT id, question, answer FROM dataset ORDER BY random() LIMIT 5;")
examples = cursor.fetchall()

results = []


print("⏱️ Préchargement du modèle LLM...")
try:
    subprocess.run(
        ["docker", "exec", "-i", "ollama", "ollama", "run", "gemma:2b"],
        input="Dis bonjour.",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120
    )
    print("✅ Modèle préchauffé")
except Exception as e:
    print(f"⚠️ Erreur pendant le préchauffage : {e}")


for i, (row_id, question, expected_answer) in enumerate(examples, 1):
    print(f"\n ({i}/5) Question (id={row_id}) : {question}")

    # Générer l'embedding de la question
    question_embedding = model.encode(question).tolist()
    print("question embedding généré")

    # top 3 documents les plus proches (selon cosine distance)
    cursor.execute(
        "SELECT context FROM dataset ORDER BY embedding <=> %s::vector LIMIT 3;",
        (question_embedding,)
    )
    context_rows = cursor.fetchall()
    context = "\n\n".join([row[0] for row in context_rows])
    print("Contexte trouvé dans bdd")

    # Construire le prompt enrichi
    enriched_prompt = f"""
    Tu es un assistant intelligent.
    Voici des documents de référence : 

    {context}

    Question : {question}
    Réponds de manière concise et précise.
    """

    # Appel LLM via Docker
    print("appel du llm")
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
            rag_answer = f"Erreur LLM : {error.strip()}"
        else:
            rag_answer = output.strip()
    except Exception as e:
        rag_answer = f"Erreur Docker : {e}"

    # Similarité cosinus
    print("calcul similarités des réponses")
    embed_expected = model.encode(expected_answer, convert_to_tensor=True)
    embed_rag = model.encode(rag_answer, convert_to_tensor=True)
    cosine_sim = util.cos_sim(embed_expected, embed_rag).item()

    print(f"Réponse attendue : {expected_answer}")
    print(f"Réponse RAG : {rag_answer}")
    print(f"Similarité cosinus : {cosine_sim:.4f}")

    results.append({
        "question_id": row_id,
        "question": question,
        "expected_answer": expected_answer,
        "rag_answer": rag_answer,
        "cosine_similarity": round(cosine_sim, 4)
    })
    print("résultats enregistrés")

# Export en CSV
output_file = "evaluation_rag_dataset.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n Évaluation terminée. Résultats exportés dans {output_file}")

cursor.close()
conn.close()
