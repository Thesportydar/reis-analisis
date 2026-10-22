"""Contrato report.data (D14): la salida de build_report_data tiene que ser igual al ejemplo que valida la app.

    python -m unittest discover -s tests
"""
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from report_data import build_report_data, DATA_VERSION  # noqa: E402

# El ejemplo vive en la app (test/fixtures); el submódulo puede estar fuera del repo padre (p. ej. en la imagen del worker)
EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "test", "fixtures", "report_data.example.json")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def frame(rows):
    return pd.DataFrame(rows, columns=["id", "pregunta", "cluster", "dup_ids"])


class ReportDataTest(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(EXAMPLE), "no está test/fixtures/report_data.example.json (submódulo fuera de la app)")
    def test_coincide_con_el_ejemplo(self):
        # q5 es casi igual a q1 y q4 a q3: el .docx las descarta, pero el análisis las incluye en el tema de su representante
        df = frame([
            ("q1", "a", 1, ["q5"]),
            ("q2", "b", 1, []),
            ("q3", "c", 2, ["q4"]),
        ])
        with open(EXAMPLE) as f:
            self.assertEqual(build_report_data(df, MODEL, 2), json.load(f))

    def test_version(self):
        self.assertEqual(DATA_VERSION, 1)

    def test_sin_preguntas(self):
        data = build_report_data(frame([]), MODEL, None)
        self.assertEqual((data["k"], data["topics"], data["duplicates"]), (0, [], []))

    def test_menos_de_tres_preguntas_es_un_tema(self):
        df = frame([("q1", "a", 1, []), ("q2", "b", 1, [])])
        data = build_report_data(df, MODEL, None)  # no se elige k
        self.assertEqual(data["k"], 1)
        self.assertEqual(data["topics"], [{"id": 1, "question_ids": ["q1", "q2"]}])

    def test_casi_igual_compartida_va_al_tema_de_la_primera(self):
        # q9 coincide con q1 (tema 1) y con q3 (tema 2): queda una sola vez, en el tema de q1
        df = frame([("q1", "a", 1, ["q9"]), ("q3", "c", 2, ["q9"])])
        data = build_report_data(df, MODEL, 2)
        self.assertEqual([t["question_ids"] for t in data["topics"]], [["q1", "q9"], ["q3"]])
        self.assertEqual(data["duplicates"], [{"ids": ["q1", "q9"]}, {"ids": ["q3", "q9"]}])

    def test_cada_pregunta_en_un_solo_tema_y_ids_como_texto(self):
        df = frame([("q1", "a", 2, ["q4"]), ("q2", "b", 1, []), ("q3", "c", 2, [])])
        data = build_report_data(df, MODEL, 2)
        ids = [i for t in data["topics"] for i in t["question_ids"]]
        self.assertEqual(sorted(ids), ["q1", "q2", "q3", "q4"])
        self.assertEqual([t["id"] for t in data["topics"]], [1, 2])
        json.dumps(data)  # serializable (los cluster de numpy se castean a int)


class PreprocessTest(unittest.TestCase):
    """`preprocess_questions` guarda los ids de las casi iguales sin cambiar qué filas quedan (embedder falso)."""

    def test_dup_ids(self):
        import numpy as np
        for name in ("sentence_transformers", "sklearn", "sklearn.cluster", "sklearn.metrics", "umap", "umap.umap_", "dotenv"):
            sys.modules.setdefault(name, types.SimpleNamespace(
                SentenceTransformer=None, KMeans=None, calinski_harabasz_score=None, UMAP=None, load_dotenv=lambda *a, **k: None))
        import QuestionClusterer as qc

        class FakeEmbedder:
            def encode(self, corpus):
                return np.array([[1.0, 0.0] if "uno" in t else [0.0, 1.0] for t in corpus])

            def similarity(self, a, b):
                return a @ b.T

        clusterer = qc.QuestionClusterer.__new__(qc.QuestionClusterer)
        clusterer.embedder = FakeEmbedder()
        rows = [("q1", "uno", "g1", "A"), ("q2", "otro", "g2", "B"), ("q3", "uno bis", "g3", "C"), ("q4", "uno ter", "g1", "A")]
        df = clusterer.preprocess_questions(rows)

        self.assertEqual(df["id"].tolist(), ["q1", "q2"])
        self.assertEqual(df["dup_ids"].tolist(), [["q3", "q4"], []])
        self.assertEqual(df["votos"].tolist(), [3, 1])
        self.assertEqual(df["grupo"].tolist(), ["g1,g3,g1", "g2"])


class ResolveDupsTest(unittest.TestCase):
    def test_cadena_no_transitiva_no_pierde_preguntas(self):
        for name in ("sentence_transformers", "sklearn", "sklearn.cluster", "sklearn.metrics", "umap", "umap.umap_", "dotenv"):
            sys.modules.setdefault(name, types.SimpleNamespace(
                SentenceTransformer=None, KMeans=None, calinski_harabasz_score=None, UMAP=None, load_dotenv=lambda *a, **k: None))
        import QuestionClusterer as qc

        # 0~1 y 1~2, pero 0 no ~ 2: la fila 1 se descarta (por la 0) y la 2 solo coincide con la 1
        matches = [[1], [0, 2], [1], []]
        self.assertEqual(qc.QuestionClusterer.resolve_dup_ids(["a", "b", "c", "d"], matches), [["b", "c"], [], [], []])


class SaveReportTest(unittest.TestCase):
    """`save_report` deja `data` en la misma sentencia que `status = 'done'`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DATABASE_URL"] = os.path.join(self.tmp.name, "test.db")
        os.environ["UPLOADS_DIR"] = self.tmp.name
        con = sqlite3.connect(os.environ["DATABASE_URL"])
        con.executescript("""
            CREATE TABLE report (
                id text PRIMARY KEY, class_id text NOT NULL, status text NOT NULL DEFAULT 'queued',
                params text, data text, error text, filename text, filepath text, mime text, score integer,
                created text NOT NULL DEFAULT current_timestamp, started text, finished text
            );
            INSERT INTO report (id, class_id, status) VALUES ('r1', 'c1', 'running');
        """)
        con.commit()
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_escribe_data_junto_con_done(self):
        for name in ("docx", "docx.shared", "docx.oxml.ns", "docx.oxml", "wordcloud", "distinctipy", "nltk", "nltk.corpus", "dotenv"):
            sys.modules.setdefault(name, types.SimpleNamespace(
                Pt=None, Inches=None, nsdecls=None, parse_xml=None, WordCloud=None, stopwords=None, load_dotenv=lambda *a, **k: None))
        try:
            import ReportGenerator as rg
        except Exception as e:  # pragma: no cover: sin dependencias de reporte no se puede importar
            self.skipTest(f"no se puede importar ReportGenerator: {e}")

        class Doc:
            def save(self, path):
                open(path, "wb").close()

        gen = rg.ReportGenerator.__new__(rg.ReportGenerator)
        gen.report_id, gen.rei_id, gen.period, gen.num_clusters = "r1", "rec", "c1", 2
        gen.document = Doc()
        gen.report_data = {"version": 1, "model": MODEL, "k": 2, "topics": [], "duplicates": []}
        rg.UPLOADS_DIR = self.tmp.name
        gen.save_report()

        con = sqlite3.connect(os.environ["DATABASE_URL"])
        status, data = con.execute("SELECT status, data FROM report WHERE id = 'r1'").fetchone()
        self.assertEqual(status, "done")
        self.assertEqual(json.loads(data)["model"], MODEL)


if __name__ == "__main__":
    unittest.main()
