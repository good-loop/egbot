#!/bin/bash

########
## Egbot - Import new CSV data into Elasticsearch
########

USAGE=$(printf "\n./add.data.sh CSV_FILE.csv\n./add.data.sh DIR_WITH_CSV_FILES/*.csv\n\n")
SERVER='winterwell@robinson.soda.sh'
TARGET='winterwell@robinson.soda.sh:/home/winterwell/egbot.good-loop.com/data/raw'

########
## Sync file(s) to the server
########
###
# Depending on the desired files to be sync'ed this could take a while.  Do this in a tmux session.
###
###
# Sub-step 01. Kill any existing uploading process that might also be running.
###
if [[ $(tmux ls | grep "egbotuploader") != '' ]]; then
    tmux kill-session -t egbotuploader
fi
###
# Sub-step 02. Create new tmux session
###
tmux new-session -d -s egbotuploader -n panel01
###
# Sub-step 03. Start the upload from within the new tmux session
###
tmux send-keys -t egbotuploader "rsync -rhP $1 $TARGET && ssh $SERVER 'cd /home/winterwell/egbot.good-loop.com && bash import.csv.sh'" C-m
###
# Sub-step 04. Print information to user and basic instructions
###
printf "\n\nTo watch the progress of your uploads:\n\ttmux attach-session -t egbotuploader\n\nTo return to this terminal session:\n\tctrl+b d\n\n"
printf "\nAfter the file(s) are uploaded, a subsequent script will be run, which will update egbot's elasticsearch with this/these new file(s).\n\n"