import os
import redis
import json
from main import main
from dotenv import load_dotenv
import logging
from MailService import send_single_email
import time

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        socket_timeout=None
    )

    r.ping()
    logging.info("Connected to Redis")

except redis.exceptions.ConnectionError as e:
    logging.error(f"Error connecting to Redis: {str(e)}")
    exit(1)


def process_task(task):
    class_id = None
    try:
        data = json.loads(task)
        class_id = data.get("class_id")

        if not class_id:
            error_msg = "Invalid task: missing class_id"
            logging.error(error_msg)
            send_single_email(
                os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"),
                "Error en Worker REI - Tarea Inválida",
                f"Error: {error_msg}\nTask data: {task}"
            )
            return

        r.set(f"task_status:{class_id}", json.dumps({"status": "in_progress"}))
        logging.info(f"Processing task for class_id: {class_id}")

        result = main(data['recorrido_id'], data['class_id'], data['email'], data['score'])

        r.set(f"task_status:{class_id}", json.dumps({
                "status": "completed",
                "download_url": result['download_url'],
                "file_name": result['file_name']
        }))
        logging.info(f"Task for class_id: {class_id} completed")

    except Exception as e:
        error_msg = f"Error procesando tarea para class_id {class_id}: {str(e)}"
        logging.error(error_msg)
        logging.exception(e)
        
        if class_id:
            r.set(
                f"task_status:{class_id}",
                json.dumps({"status": "failed", "error": str(e)}),
                ex=900 # 15 minutos
            )
        
        # Enviar notificación al administrador
        try:
            send_single_email(
                os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"),
                "Error en Worker REI - Fallo en Procesamiento",
                f"{error_msg}\n\nDetalles de la tarea:\n{json.dumps(data if class_id else {'raw': str(task)}, indent=2)}"
            )
        except Exception as email_error:
            logging.error(f"No se pudo enviar email de notificación al admin: {email_error}")
        
        raise

if __name__ == '__main__':
    print("Worker started...")

    while True:
        try:
            task = r.brpop('tasks', timeout=0)
            if task:
                process_task(task[1])

        except redis.exceptions.ConnectionError as e:
            error_msg = f"Se perdió la conexión con Redis: {e}. Intentando reconectar..."
            logging.error(error_msg)
            
            # Notificar al admin sobre la pérdida de conexión
            try:
                send_single_email(
                    os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"),
                    "Worker REI - Pérdida de Conexión Redis",
                    error_msg
                )
            except Exception as email_error:
                logging.error(f"No se pudo enviar email de notificación al admin: {email_error}")
            
            time.sleep(5)  # Espera antes de intentar reconectar
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                socket_timeout=None
            )

        except Exception as e:
            error_msg = f"Error inesperado en el worker: {e}"
            logging.error(error_msg)
            logging.exception(e)
            
            # Notificar al admin sobre errores inesperados
            try:
                send_single_email(
                    os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"),
                    "Worker REI - Error Crítico",
                    f"{error_msg}\n\nEl worker continuará procesando otras tareas."
                )
            except Exception as email_error:
                logging.error(f"No se pudo enviar email de notificación al admin: {email_error}")
            
            continue
