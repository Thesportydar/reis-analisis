# Submódulo de análisis IA para Reis App

Este submódulo está diseñado para realizar tareas de análisis dentro de la aplicación Reis. A continuación, se detallan los pasos para configurar el entorno y comenzar a trabajar.

## Configuración del Entorno

1. **Crear un entorno virtual**  
   Ejecuta el siguiente comando para crear un entorno virtual en el directorio del proyecto:
   ```sh
   python3 -m venv .venv
   ```

2. **Activar el entorno virtual**
    - En macOS/Linux:
    ```sh
    source .venv/bin/activate
    ```
    - En Windows:
    ```cmd
    .venv\Scripts\activate
    ```

3. **Instalar dependencias**
    Una vez activado el entorno virtual, instala las dependencias necesarias ejecutando:
    ```bash
    pip install -r requirements.txt
    ```

## Configuración de Variables de Entorno
El archivo `.env` contiene las variables de entorno necesarias para el funcionamiento del submódulo. Estas incluyen configuraciones como el host de Redis, la base de datos, y claves API. Asegúrate de revisar y ajustar estas variables según sea necesario.

Sin embargo, si solo vas a realizar tareas de exploración con las notebooks, no es obligatorio configurar estas variables.

## Uso de las Notebooks
Para cargar preguntas de ejemplo y trabajar con la base de datos, utiliza la notebook `Load_db`. Si no existe una base de datos, esta notebook creará una por defecto.

## Notas Adicionales
- Asegúrate de que las dependencias estén correctamente instaladas antes de ejecutar cualquier notebook o script.
- Si tienes dudas sobre la configuración de las variables de entorno, consulta el archivo .env incluido en este submódulo.

¡Listo! Ahora puedes comenzar a trabajar con el submódulo de análisis IA para Reis App.