# coding=utf-8
import os
import tensorflow as tf
from flask import Flask, request,render_template,jsonify
from werkzeug import secure_filename
from PIL import Image
import numpy as np
import time
from cassandra.cluster import Cluster

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG','jpeg','JPEG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


sess = tf.Session()
saver = tf.train.import_meta_graph("./checkpoint/model.ckpt.meta")
saver.restore(sess, './checkpoint/model.ckpt')
keep_prob = tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
y_conv = tf.get_default_graph().get_tensor_by_name('fc2/add:0')


cluster = Cluster(['127.0.0.1'])
session = cluster.connect()
session.execute("create KEYSPACE mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("create table images(id uuid, digits int, upload_image list<int>, upload_time timestamp,primary key(id));")


def imageprepare(argv):
    im = Image.open(argv)
    imout = im.convert('L')
    xsize, ysize = im.size
    if xsize != 28 or ysize != 28:
        imout = imout.resize((28, 28), Image.ANTIALIAS)
        imout.save("return.png", "png")
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(1.0 - float(imout.getpixel((j, i))) / 255.0)
            arr.append(pixel)
    return arr


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    f = request.files['file']  
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        array = imageprepare(filename)
        prediction = tf.argmax(y_conv, 1)
        y_pre = prediction.eval(feed_dict={x: [array], keep_prob: 1.0}, session=sess)
        uploadtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        session.execute("INSERT INTO images(id, digits, upload_image, upload_time) values(uuid(), y_pre[0], array, uploadtime);")
    return jsonify({'The digits in this image is':str(y_pre[0])})


if __name__ == '__main__':
    app.run()
