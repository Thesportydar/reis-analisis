"""Tests de la cola de reportes del worker (D14): reclamo atómico, reencolado y despertar por FIFO.

Corren sin las dependencias pesadas (torch, etc.): `main` y `MailService` se reemplazan por stubs.
    python -m unittest discover -s tests
"""
import os
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_calls = []
sys.modules["main"] = types.SimpleNamespace(main=lambda *a, **k: _calls.append((a, k)))
sys.modules["MailService"] = types.SimpleNamespace(send_single_email=lambda *a, **k: None)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

import worker  # noqa: E402


class WorkerQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DATABASE_URL"] = os.path.join(self.tmp.name, "test.db")
        con = sqlite3.connect(os.environ["DATABASE_URL"])
        con.executescript("""
            CREATE TABLE class (id text PRIMARY KEY, recorrido_id text);
            CREATE TABLE report (
                id text PRIMARY KEY, class_id text NOT NULL, status text NOT NULL DEFAULT 'queued',
                params text, error text, filename text, filepath text, mime text, score integer,
                created text NOT NULL DEFAULT current_timestamp, started text, finished text
            );
            INSERT INTO class VALUES ('c1', 'r1');
        """)
        con.commit()
        con.close()
        _calls.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def insert(self, id, status="queued", created="2026-01-01 00:00:00"):
        con = worker.connect()
        con.execute("INSERT INTO report (id, class_id, status, params, created) VALUES (?, 'c1', ?, '{\"email\": \"a@b.c\", \"score\": 2}', ?)",
                    (id, status, created))
        con.commit()
        con.close()

    def status(self, id):
        con = worker.connect()
        row = con.execute("SELECT status, started, error FROM report WHERE id = ?", (id,)).fetchone()
        con.close()
        return row

    def test_claim_toma_el_mas_viejo_una_sola_vez(self):
        self.insert("b", created="2026-01-02 00:00:00")
        self.insert("a", created="2026-01-01 00:00:00")
        first = worker.claim_next()
        self.assertEqual(first[0], "a")
        self.assertEqual(self.status("a")[0], "running")
        self.assertIsNotNone(self.status("a")[1])
        self.assertEqual(worker.claim_next()[0], "b")
        self.assertIsNone(worker.claim_next())

    def test_requeue_stale_vuelve_a_encolar_los_running(self):
        self.insert("a", status="running")
        self.insert("b", status="done")
        worker.requeue_stale()
        self.assertEqual(self.status("a")[0], "queued")
        self.assertEqual(self.status("b")[0], "done")

    def test_process_task_pasa_los_params_y_resuelve_el_recorrido(self):
        self.insert("a")
        worker.process_pending()
        (args, kwargs), = _calls
        self.assertEqual(args, ("r1", "c1", "a@b.c", 2))
        self.assertEqual(kwargs, {"report_id": "a"})

    def test_fallo_deja_failed_con_error(self):
        self.insert("a")
        original = worker.main
        def boom(*a, **k): raise Exception("se rompió")
        worker.main = boom
        try:
            worker.process_pending()
        finally:
            worker.main = original
        status, _, error = self.status("a")
        self.assertEqual((status, error), ("failed", "se rompió"))

    def test_fifo_despierta_al_worker(self):
        fd = worker.open_wake()
        try:
            threading.Timer(0.2, lambda: os.write(os.open(worker.wake_path(), os.O_WRONLY | os.O_NONBLOCK), b"1")).start()
            start = time.time()
            worker.wait_for_wake(fd, timeout=5)
            self.assertLess(time.time() - start, 2)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
