from flask import Flask, render_template, request, redirect, flash, url_for
import prediction_api
import urllib.request
from app import app
from werkzeug.utils import secure_filename
from prediction_api import getPrediction
import os
import  matplotlib.pyplot as plt
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # im =np.array(cv2.imread(filename), dtype=float)
            # plt.imshow(im)
            label, acc = getPrediction(filename)
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')
# @app.route('/', methods=['POST'])
# def submit_file():
#     filename = take_pic()
#
#     label, acc = getPrediction(filename)
#     flash(label)
#     flash(acc)
#     flash(filename)
#     return redirect('/')

# def take_pic():
#     cam = cv2.VideoCapture(0)
#
#     cv2.namedWindow("Take a bird's picture!")
#
#     img_counter = 0
#
#     while True:
#
#         ret, frame = cam.read()
#
#         if not ret:
#             print("Failed to grab a frame")
#             break
#
#         cv2.imshow("Take a bird's picture!", frame)
#         k = cv2.waitKey(1)
#
#         if k % 256 == 27:
#             # ESC pressed
#             print("Escape hit, closing camera!")
#             break
#
#         elif k % 256 == 32:
#             # SPACE pressed
#             img_name = f"bird_img_{img_counter}.png"
#             cv2.imwrite('data/Predicions/'+img_name, frame)
#             print(f"{img_name} written! You may take another one OR exit by \
#     hitting ESC")
#             img_counter += 1
#
#     cam.release()
#     cv2.destroyAllWindows()
#     return img_name

if __name__ == "__main__":
    app.run()