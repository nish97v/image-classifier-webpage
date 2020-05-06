import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_file
from utils import is_allowed_file
from werkzeug.utils import secure_filename

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
            return redirect(url_for('predict', filename='test.png'))
        else:
            # Need to do : Return the type of error
            flash('An error occured, try again.')
            return redirect(request.url)


# Angular bracket for filename is used to get filename as a parameter
@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    # Need to do : logic for loading the uploaded image file and predict the class

    return render_template('predict.html')


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(app.config['UPLOAD_FOLDER'], f'images/{filename}')


if __name__ == "__main__":
    app.run('127.0.0.1')
