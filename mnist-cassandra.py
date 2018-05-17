# coding=utf-8
import os
import tensorflow as tf
from flask import Flask, request,render_template,jsonify
from werkzeug import secure_filename
import time
from cassandra.cluster import Cluster
from convert_pic import *
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


cluster = Cluster(['172.17.0.1'],port=9042)
session = cluster.connect()
session.execute("create KEYSPACE if not exists mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("create table if not exists ABCDE(id uuid, digits int,image_name text, upload_time timestamp, primary key(id));")



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('upload.html')



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
        session.execute("INSERT INTO ABCDE(id, digits, image_name , upload_time) values(uuid(), %s, %s, %s)",[y_pre[0], filename, uploadtime])
        return jsonify({'The digits in this image is':str(y_pre[0])})


if __name__ == '__main__':
    app.run(port=5000)
