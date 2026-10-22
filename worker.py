import os
import json
import select
import logging
from contextlib import closing
from main import main
from dotenv import load_dotenv
from MailService import send_single_email
from db import connect

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# La cola es la tabla `report` (D14). El FIFO solo despierta al worker: si el aviso se pierde,
# el chequeo de respaldo (CHECK_INTERVAL) encuentra igual la fila en cola.
CHECK_INTERVAL = 60


def wake_path():
    return f"{os.getenv('DATABASE_URL', 'sqlite.db')}.wake"


def open_wake():
    """Crea el FIFO junto a la DB (si falta) y lo abre lectura/escritura: así nunca ve EOF aunque no haya escritores."""
    path = wake_path()
    if not os.path.exists(path):
        os.mkfifo(path)
    return os.open(path, os.O_RDWR | os.O_NONBLOCK)


def drain(fd):
    try:
        while os.read(fd, 4096):
            pass
    except BlockingIOError:
        pass


def wait_for_wake(fd, timeout=CHECK_INTERVAL):
    select.select([fd], [], [], timeout)
    drain(fd)


def requeue_stale():
    """Un worker que murió a mitad de un trabajo deja filas `running`: se vuelven a encolar."""
    with closing(connect()) as conn:
        cur = conn.execute("UPDATE report SET status = 'queued', started = NULL WHERE status = 'running'")
        conn.commit()
        if cur.rowcount:
            logging.warning(f"{cur.rowcount} trabajo(s) quedaron en curso: se vuelven a encolar")


def claim_next():
    """Toma el trabajo más viejo en cola (atómico). Devuelve (id, class_id, params) o None."""
    with closing(connect()) as conn:
        row = conn.execute(
            """
            UPDATE report SET status = 'running', started = current_timestamp
            WHERE id = (SELECT id FROM report WHERE status = 'queued' ORDER BY created, rowid LIMIT 1)
            RETURNING id, class_id, params
            """
        ).fetchone()
        conn.commit()
        return row


def fail(report_id, error):
    with closing(connect()) as conn:
        conn.execute(
            "UPDATE report SET status = 'failed', error = ?, finished = current_timestamp WHERE id = ?",
            (error, report_id)
        )
        conn.commit()


def notify_admin(subject, body):
    try:
        send_single_email(os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"), subject, body, use_admin_key=True)
    except Exception as email_error:
        logging.error(f"No se pudo enviar email de notificación al admin: {email_error}")


def process_task(report_id, class_id, params):
    try:
        data = json.loads(params) if params else {}
        logging.info(f"Processing report {report_id} for class_id: {class_id}")

        with closing(connect()) as conn:
            row = conn.execute("SELECT recorrido_id FROM class WHERE id = ?", (class_id,)).fetchone()
        if not row:
            raise Exception(f"La sesión {class_id} no existe")

        main(row[0], class_id, data.get("email"), data.get("score", 0), report_id=report_id)
        logging.info(f"Report {report_id} for class_id: {class_id} completed")

    except Exception as e:
        error_msg = f"Error procesando reporte {report_id} (class_id {class_id}): {str(e)}"
        logging.error(error_msg)
        logging.exception(e)
        fail(report_id, str(e))
        notify_admin(
            "Error en Worker REI - Fallo en Procesamiento",
            f"{error_msg}\n\nDetalles de la tarea:\n{json.dumps({'report_id': report_id, 'class_id': class_id, 'params': params}, indent=2)}"
        )


def process_pending():
    while (task := claim_next()):
        process_task(*task)


if __name__ == '__main__':
    print("Worker started...")

    wake_fd = open_wake()
    requeue_stale()

    while True:
        try:
            process_pending()
            wait_for_wake(wake_fd)
        except Exception as e:
            error_msg = f"Error inesperado en el worker: {e}"
            logging.error(error_msg)
            logging.exception(e)
            notify_admin("Worker REI - Error Crítico", f"{error_msg}\n\nEl worker continuará procesando otras tareas.")
            wait_for_wake(wake_fd, 5)
