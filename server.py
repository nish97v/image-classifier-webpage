import os
import numpy as np
from flask import Flask, flash, render_template, request, redirect, url_for, send_file
from utils import check_image_dir, is_allowed_file, generate_barplot
from werkzeug.utils import secure_filename
from test import load_and_preprocess, predict_probabilities

production = False

app = Flask(__name__)
if production:
    app.config.from_object('config.ProductionConfig')
else:
    app.config.from_object('config.Config')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        passed = False
        if image_file and is_allowed_file(image_file.filename, app.config['ALLOWED_EXTENSIONS']):
            try:
                filename = secure_filename(image_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'images/{filename}')
                image_file.save(file_path)
                passed = True
            except Exception:
                passed = False

        if passed:
            return redirect(url_for('predict', filename=image_file.filename))
        else:
            # Need to do : Return the type of error
            flash('An error occurred, try again.')
            return redirect(request.url)


# Angular bracket for filename is used to get filename as a parameter
@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    image_url = url_for('images', filename=filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'images/{filename}')
    image = load_and_preprocess(image_path)
    probabilities = predict_probabilities(image)
    prediction = app.config['IMAGE_LABELS'][np.where(probabilities > 0.5)[0][0]]
    script, div = generate_barplot(probabilities, app.config['IMAGE_LABELS'])
    return render_template('predict.html', plot_script=script, plot_div=div, image_url=image_url,
                           prediction=prediction)


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], f'images/{filename}'))


if __name__ == "__main__":
    check_image_dir()
    app.run('127.0.0.1')
