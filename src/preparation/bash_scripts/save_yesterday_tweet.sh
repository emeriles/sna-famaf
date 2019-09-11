#!/usr/bin/env bash

##########
# mongo container should be created first by running:
# cd PROJECT_DIR
# docker create --name db_tweets --restart on-failure -p 27017:27017 -v db:/data/db mongo
##########


NAME="save_tweets_yesterday"
PROJECT_DIR=/home/ubuntu/projects/emanuel/sna-famaf/src/preparation
VIRTUAL_ENV=/home/ubuntu/.virtualenvs/sna-famaf

source $VIRTUAL_ENV/bin/activate

DATE=`date +%Y-%m-%d`

cd $PROJECT_DIR
docker start db_tweets
exec $VIRTUAL_ENV/bin/python -u actions.py get_yesterday 2>&1 | tee actions_log_yesterday_$DATE.txt
docker stop db_tweets

# for crontab. at 2:00 am.
# 0 2 * * * /home/ubuntu/projects/emanuel/sna-famaf/tw_dataset/bash_scripts/save_yesterday_tweet.sh