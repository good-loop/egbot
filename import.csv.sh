#!/bin/bash

#####
## EgBot server : Process new CSV
#####

####
## Kill any existing prior "Process-new-CSV" Tmux session that might be running
####
if [[ $(tmux ls | grep egbotcsvimport) != '' ]]; then
    tmux kill-session -t egbotcsvimport
fi

####
## Create new Tmux session
####
tmux new-session -d -s egbotcsvimport -n panel01

####
## Start the Java Process which translates/imports the CSV files into elasticsearch
####
tmux send-keys -t egbotcsvimport "java -cp lib/*: com.goodloop.egbot.tools.Step2_CSVIntoES" C-m

####
## Print how-to-view progress and basic instructions for detaching from tmux
####
printf "\n###If this script was auto-run from 'upload.csv.sh',\n\tthen you will need to ssh into robinson in order to view importing progress.\n\nOtherwise\n\nTo view the progress of the CSV importing:\n\ttmux attach-session -t egbotcsvimport\n\nTo detach from viewing the progress, and return to this terminal:\n\tctrl+b d\n\n"
