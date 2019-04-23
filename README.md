# sna-famaf
Social Network Análisis on Twitter: Early trending tweet prediction. It consist of the final project for Computer Scientist grade degree.


# Installation

```
mkvirtualenv -p /usr/bin/python3.6 sna-famaf
pip install -r requirements.txt
```


# Usefull stuff

export dataset as csv
`mongoexport -h localhost -d twitter -c tweet --type=csv --fields created_at,user.id_str,id_str,text,retweeted_status.id_str,retweeted_status.user.id_str,retweeted_status.created_at,retweet_count,quoted_status_id_str --out dayli_col.csv`
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
