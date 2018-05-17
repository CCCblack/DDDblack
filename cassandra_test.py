from cassandra.cluster import Cluster
import time

cluster = Cluster(['172.17.0.1'],port=9042)
session = cluster.connect()
session.execute("create KEYSPACE if not exists mnist_database WITH replication = {'class':'SimpleStrategy', 'replication_factor': 2};")
session.execute("use mnist_database")
session.execute("DROP TABLE if exists images")
session.execute("create table if not exists images(id uuid, digits int, upload_image list<float>, upload_time timestamp, primary key(id));")

y_pre=1
array = [1.0,2.0]
uploadtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
session.execute("INSERT INTO images(id, digits, upload_image, upload_time) values(uuid(), %s, %s, %s)",[y_pre, array, uploadtime])
rows = session.execute('SELECT * FROM images')
for user_row in rows:
    print user_row.id, user_row.digits, user_row.upload_image, user_row.upload_time
hei = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print hei
