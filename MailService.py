import json
import requests
import logging

import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

# API key para emails a usuarios (reportes)
API_KEY = os.getenv("MAILGUN_API_KEY", '')

# API key para emails administrativos (notificaciones de errores)
ADMIN_API_KEY = os.getenv("MAILGUN_ADMIN_API_KEY", API_KEY)  # Si no existe, usa la misma que la normal

MAILGUN_API_URL = "https://api.mailgun.net/v3/reports.inaqui.me/messages"
FROM_EMAIL_ADDRESS = "REI <mailgun@reports.inaqui.me>"

def send_single_email(to_address: str, subject: str, message: str, use_admin_key: bool = False):
    """
    Envía un email simple.
    
    Args:
        to_address: Dirección de correo del destinatario
        subject: Asunto del email
        message: Cuerpo del mensaje
        use_admin_key: Si es True, usa MAILGUN_ADMIN_API_KEY en lugar de MAILGUN_API_KEY
    """
    api_key = ADMIN_API_KEY if use_admin_key else API_KEY
    
    try:
        resp = requests.post(
            MAILGUN_API_URL,
            auth=("api", api_key),
            data={
                "from": FROM_EMAIL_ADDRESS,
                "to": to_address,
                "subject": subject,
                "text": message
            }
        )

        if resp.status_code == 200:
            logging.info(f"Successfully sent an email to '{to_address}' via Mailgun API.")
        else:
            logging.error(f"Could not send the email, reason: {resp.text}")

    except Exception as ex:
        logging.exception(f"Mailgun error: {ex}")


def send_email_with_attachment(to_address: str, subject: str, message: str, attachment_path: str, attachment_name: str):
    """
    Envía un email con archivo adjunto (siempre usa la API key normal para reportes a usuarios).
    """
    try:
        # files = {'attachment': open(attachment_path, 'rb')}
        with open(attachment_path, 'rb') as attachment:
            # le cambiamos el nombre al archivo adjunto
            files = {'attachment': (attachment_name, attachment)}

            resp = requests.post(
                MAILGUN_API_URL,
                auth=("api", API_KEY),
                files=files,
                data={
                    "from": FROM_EMAIL_ADDRESS,
                    "to": to_address,
                    "subject": subject,
                    "text": message
                }
            )

            if resp.status_code == 200:
                logging.info(f"Successfully sent an email to '{to_address}' via Mailgun API.")
            else:
                logging.error(f"Could not send the email, reason: {resp.text}")

    except Exception as ex:
        logging.exception(f"Mailgun error: {ex}")


if __name__ == "__main__":
    send_single_email("Inaqui <ipaladinobravo@hotmail.com>", "Single email test", "Testing Mailgun API for a single email")
