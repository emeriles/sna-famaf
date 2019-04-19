#!/usr/bin/env bash

NAME="save_tweets_yesterday"
PROJECT_DIR=/home/ubuntu/projects/emanuel/sna-famaf/tw_dataset/
VIRTUAL_ENV=/home/ubuntu/.virtualenvs/sna-famaf

source $VIRTUAL_ENV/bin/activate

DATE=`date +%Y-%m-%d`

cd $PROJECT_DIR
exec $VIRTUAL_ENV/bin/python -u actions.py get_yesterday 2>&1 | tee actions_log_yesterday_$DATE.txt

# for crontab. at 2:00 am.
# 0 2 * * * /home/ubuntu/projects/emanuel/sna-famaf/tw_dataset/bash_scripts/save_yesterday_tweet.sh
