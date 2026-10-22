"""Contrato `report.data` (D14): lo que el worker le deja a la app. Solo IDs, sin textos.

    { version, model, k, topics: [{ id, question_ids }], duplicates: [{ ids }] }

Lo valida `src/schemas/reportData.ts` en la app: si cambia algo acá, subir DATA_VERSION y actualizar el esquema.
"""

DATA_VERSION = 1


def build_report_data(df, model, k):
    """Arma el contrato desde el DataFrame de `QuestionClusterer.clusterize_dx_questions`.

    `df` tiene una fila por pregunta representante, con `id`, `cluster` y `dup_ids` (las casi iguales que el
    `.docx` descarta). Cada casi igual cae en el tema de la primera representante que la reclama.
    """
    topics = {}
    duplicates = []
    placed = set()

    for row in df.itertuples():
        cluster = row.cluster
        if cluster is None or cluster != cluster:  # None / NaN: sin clusterizar
            continue
        topic = topics.setdefault(int(cluster), [])
        for question_id in [row.id, *row.dup_ids]:
            if question_id not in placed:
                placed.add(question_id)
                topic.append(question_id)
        if row.dup_ids:
            duplicates.append({"ids": [row.id, *row.dup_ids]})

    return {
        "version": DATA_VERSION,
        "model": model,
        "k": (int(k) if k else len(topics)) if topics else 0,  # con menos de 3 preguntas no se elige k: hay un solo tema
        "topics": [{"id": cluster, "question_ids": ids} for cluster, ids in sorted(topics.items())],
        "duplicates": duplicates,
    }
