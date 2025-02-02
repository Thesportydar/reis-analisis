from QuestionClusterer import QuestionClusterer
from ReportGenerator import ReportGenerator
import sys
import json
from MailService import send_email_with_attachment


def main(rei_id, period, email, score=0):
    # Clusterizar preguntas
    clusterer = QuestionClusterer(score)
    # Dx questions
    df_info_reis_pregs = clusterer.clusterize_dx_questions(rei_id, period)
    # New questions
    df_info_reis_pregnuevas = clusterer.clusterize_new_questions(rei_id, period)

    # Generar reporte
    report_generator = ReportGenerator(
        rei_id,
        period,
        df_info_reis_pregs,
        df_info_reis_pregnuevas,
        clusterer.num_clusters
    )

    report = report_generator.generate_report()


    # Enviar correo con el reporte adjunto
    send_email_with_attachment(
        email,
        f"REIS: Reporte Recorrido {rei_id} Clase {period}",
        "Reporte generado por el sistema REI",
        report.get('file_path', ''),
        report.get('file_name', '')
    )

    return report


if __name__ == '__main__':
    data = json.loads(sys.stdin.read())  # Lee datos de entrada
    rei_id = data.get('recorrido_id')
    period = data.get('class_id')
    email = data.get('email')
    main(rei_id, period, email)
