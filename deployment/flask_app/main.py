import os
from pathlib import Path

import classify
from flask import Flask, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "./downloads"
ALLOWED_EXTENSIONS = {"txt", "csv", "tsv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.abspath(UPLOAD_FOLDER)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

classify.load_artifacts()


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


@app.route("/info", methods=["GET"])
def info():
    return {
        "status": "ok",
        "data": {
            "task": "ecg_classification",
            "metadata": classify.metadata,
            "preprocessing": classify.preprocessing,
        },
    }


@app.route("/classify-text", methods=["POST"])
def classify_text():
    try:
        signal_text = request.form.get("signal", "")
        signal = classify.parse_signal_text(signal_text)
        result = classify.classify_signal(signal)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/classify-file", methods=["POST"])
def classify_file():
    try:
        if "signal_file" not in request.files:
            return {"status": "error", "message": "Fichier manquant"}, 400

        signal_file = request.files["signal_file"]
        if not signal_file or signal_file.filename == "":
            return {"status": "error", "message": "Fichier vide"}, 400
        if not allowed_file(signal_file.filename):
            return {"status": "error", "message": "Extension invalide (.txt, .csv ou .tsv attendue)"}, 400

        filename = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(signal_file.filename))
        signal_file.save(filename)

        signal = classify.parse_signal_file(filename)
        result = classify.classify_signal(signal)
        return {"status": "ok", "data": result, "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/", methods=["GET"])
def home():
    return '''
    <!doctype html>
    <html>
        <head>
            <title>ECG Flask API</title>
            <meta charset="UTF-8" />
        </head>
        <body>
            <h1>ECG Flask API</h1>
            <p>API prête.</p>
            <ul>
                <li>GET /health</li>
                <li>GET /info</li>
                <li>POST /classify-text</li>
                <li>POST /classify-file</li>
            </ul>
        </body>
    </html>
    '''


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, use_reloader=False)
