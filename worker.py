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
    try:
        data = json.loads(task)
        class_id = data.get("class_id")

        if not class_id:
            logging.error("Invalid task: missing class_id")
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
        r.set(
            f"task_status:{class_id}",
            json.dumps({"status": "failed", "error": str(e)}),
            ex=900 # 15 minutos
        )
        raise

if __name__ == '__main__':
    print("Worker started...")

    while True:
        try:
            task = r.brpop('tasks', timeout=0)
            if task:
                process_task(task[1])

        except redis.exceptions.ConnectionError as e:
            logging.error(f"Se perdió la conexión con Redis: {e}. Intentando reconectar...")
            time.sleep(5)  # Espera antes de intentar reconectar
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                socket_timeout=None
            )

        except Exception as e:
            logging.error(f"Error procesando tarea: {e}")
            logging.exception(e)
            continue
            # send_single_email(
                # os.getenv("ADMIN_EMAIL", "ipaladinobravo@gmail.com"),
                # "El Worker de REI se ha detenido",
                # f"Error procesando tarea: {e}"
            # )
            # exit(1)
