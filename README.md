# sna-famaf
Social Network Análisis on Twitter: Early trending tweet prediction. It consist of the final project for Computer Scientist grade degree.


# Installation

```
mkvirtualenv -p /usr/bin/python3.6 sna-famaf
pip install -r requirements.txt
```


# Usefull stuff

Export dataset as csv.
(as it is really hard to work with text on csvs, you may want to omit text field)
`mongoexport -h localhost -d twitter -c tweet --type=csv --fields created_at,user.id_str,id_str,text,retweeted_status.id_str,retweeted_status.user.id_str,retweeted_status.created_at,retweet_count,quoted_status_id_str --out dayli_col.csv`

Export text fiedls to pandas dataframe:
put mongo docker up, then
`ipython` from src
```python
from preparation.db import DBHandler 
import pandas as pd
dbh = DBHandler()
r = dbh.tweet_collection.find({}, {'id_str':1, 'text': 1, '_id': 0, 'retweeted_status.id_str': 1, 'retweeted_status.text': 1})
df = pd.DataFrame(list(r))

```
\
\
\
\
\
\
\
\
\
\

Notas:


tiempo para colectar tweets del dia: 61.1 minutos.
tiempo para colectar todos tweets posibles 3 a 5 dias.

tamaño db: 205 mb (jueves 11 de octubre 13:hs) (182005 tweets + users)
tamaño db: 257 mb (jueves 11 de octubre 15:hs) (226777 tweets + users)
tamaño db: 320 mb (vierne 12 de octubre 14:hs) (281336 tweets + users)
tamaño db: 379 mb (vierne 13 de octubre 14:hs) (331284 tweets + users)
tamaño db: 465 mb (luness 15 de octubre 20:hs) (405565 tweets + users)
tamaño db: 518 mb (martes 16 de octubre 20:hs) (451031 tweets + users)
tamaño db: 577 mb (martes 16 de octubre 20:hs) (502843 tweets + users)
tamaño db: 643 mb (jueves 18 de octubre 10:hs) (558950 tweets + users)
tamaño db: 854 mb (luness 22 de octubre 10:hs) (740397 tweets + users)


db pt1:
count: 7706798
created at:
 'max': 'Wed Sep 30 23:59:57 +0000 2015',
 'min': 'Fri Apr 01 00:00:47 +0000 2016'


db pt2:
count: 7747919
created at:
 'max': 'Wed Sep 30 23:59:42 +0000 2015'
 'min': 'Fri Apr 01 00:00:01 +0000 2016'

db pt2: 
count: 171128
created at:
 'max': 'Wed Sep 30 23:02:47 +0000 2015'
 'min': 'Fri Apr 01 00:04:17 +0000 2016'


db full:
count:  15462707  (faltan 7990 al sumar pt1 + pt2) 
count:  15633833
created at:
 'max': ''
 'min': ''



En dataset de juguete:
    antes del reciclaje:
    8% RT; 91% originales
    despues del reciclaje:
    16% RT; 83% originales


daily make cut1 (?)
real	211m16.106s
user	188m41.022s
sys	2m55.341s


Cant lineas 
14207279 cut1_full_col.csv
18314824 full.csv


create and save csv cutted full:
real	60m4.848s
user	55m58.257s
sys	4m7.879s


migrate csv cutted to sqlite (**DEPRACATED**)
real    2413m28.159s
user    2248m50.022s
sys     76m44.152s


FEATURES MATIX:
    dayli: Extracting features. X shape is :                       (123824, 5173)
    full: Extracting features. X shape is :                      (12219685, 5173)
          Extracting features Optimized. X shape is :             (9087871, 3572)

Dataframe loading:
    full:
        Done loading df. DF shape is :(11902427, 8) (Original: (12397359, 8))           Time delta is: 75 mins
                                                               (12702867, 2)
        
        12219685
        11902427


time python main.py --data full compute_scores 75
Done loading df. DF shape is :(11902427, 8) (Original: (12397359, 8))           Time delta is: 75 mins
done getting tweets universe. Shape is  (4786053, 2)
X SIZE (MB):  24241358445000000
Extracting features Optimized. X shape is : (4786053, 5065)
Avance: %0.00190135796657386935
	real    697m19.818s
	user    691m35.054s
	sys     6m39.975s
     
   
## SPARK

to install it took to follow this guide:
https://medium.com/datos-y-ciencia/c%C3%B3mo-usar-pyspark-en-tu-computadora-dee7978d1a40

and to add to bashrc:
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

export SPARK_HOME=/home/mmeriles/spark/spark-2_4_3-bin-hadoop2_7
export PATH=$SPARK_HOME/bin:$PATH

export PYSPARK_DRIVER_PYTHON=jupyter
```

test example:

```python
import findspark
import os
findspark.init(os.environ.get('SPARK_HOME'))
import random
from pyspark import SparkContext
sc = SparkContext(appName="EstimatePi")
def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1
NUM_SAMPLES = 1000000
count = sc.parallelize(range(0, NUM_SAMPLES)) \
             .filter(inside).count()
print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))
sc.stop()
```
