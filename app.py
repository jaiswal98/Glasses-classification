from flask import Flask, render_template, request, send_file, url_for
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil
import zipfile
import uuid
import threading
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model
import os
import gdown

model_path = "glasses_classifier.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1JF8N1q2bwqDkAdpBk-dUojD7ZUSm2EqA"  # not the full sharing link
    gdown.download(url, "glasses_classifier.h5", quiet=False)

try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading model:", e)
    model = None
if model is None:
    return "Model not loaded", 500

input_shape = model.input_shape[1:3]  # e.g., (256, 256)

# Directory for storing results
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def prepare_image(image_path):
    """Prepare image for model prediction"""
    img = load_img(image_path, target_size=input_shape)
    img = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)


def schedule_deletion(folder_path, delete_after_days=30):
    """Schedule folder deletion after a given time (default 1 month)"""
    delete_time = datetime.now() + timedelta(days=delete_after_days)

    def delete_folder():
        while datetime.now() < delete_time:
            time.sleep(3600)  # Check every hour
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    threading.Thread(target=delete_folder, daemon=True).start()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')

        session_id = str(uuid.uuid4())  # Unique session ID
        session_folder = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        glasses_path = os.path.join(session_folder, 'glasses')
        noglasses_path = os.path.join(session_folder, 'noglasses')

        os.makedirs(glasses_path, exist_ok=True)
        os.makedirs(noglasses_path, exist_ok=True)

        predictions = []

        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process and predict
            img = prepare_image(filepath)
            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                label = 'glasses'
                dest_folder = glasses_path
            else:
                label = 'noglasses'
                dest_folder = noglasses_path

            shutil.copy(filepath, os.path.join(dest_folder, filename))
            predictions.append((filename, label))
            os.remove(filepath)  # Cleanup temp file

        # Zip Results
        zip_filename = f"{session_id}_results.zip"
        zip_filepath = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for root, _, files in os.walk(session_folder):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), session_folder))

        # Schedule deletion of session folder after 1 month
        schedule_deletion(session_folder)

        return render_template(
            'index.html',
            predictions=predictions,
            download_link=url_for('download', zip_file=zip_filename)
        )

    return render_template('index.html')


@app.route('/download/<zip_file>')
def download(zip_file):
    """Serve the ZIP file for download"""
    zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_file)
    return send_file(zip_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
