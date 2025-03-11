from QuestionClusterer import QuestionClusterer
from ReportGenerator import ReportGenerator
import sys
import json
from MailService import send_email_with_attachment


def main(rei_id, period, email, score=0):
    # Clusterizar preguntas
    clusterer = QuestionClusterer(score)

    # Dx questions
    try:
        df_info_reis_pregs = clusterer.clusterize_dx_questions(rei_id, period)
    except Exception as e:
        raise Exception(f"Error clusterizando preguntas dx: {str(e)}")


    # New questions
    try:
        df_info_reis_pregnuevas = clusterer.clusterize_new_questions(rei_id, period)
    except Exception as e:
        raise Exception(f"Error clusterizando preguntas nuevas: {str(e)}")


    # Generar reporte
    try:
        report_generator = ReportGenerator(
            rei_id,
            period,
            df_info_reis_pregs,
            df_info_reis_pregnuevas,
            clusterer.num_clusters
        )
        report = report_generator.generate_report()

    except Exception as e:
        raise Exception(f"Error generando reporte: {str(e)}")


    # Enviar correo con el reporte adjunto
    try:
        send_email_with_attachment(
            email,
            f"REI: Reporte Recorrido {rei_id} Clase {period}",
            "Reporte generado por el sistema REI",
            report.get('file_path', ''),
            report.get('file_name', '')
        )

    except Exception as e:
        raise Exception(f"Error enviando correo con reporte adjunto: {str(e)}")


    return report


if __name__ == '__main__':
    data = json.loads(sys.stdin.read())  # Lee datos de entrada
    rei_id = data.get('recorrido_id')
    period = data.get('class_id')
    email = data.get('email')
    main(rei_id, period, email)
