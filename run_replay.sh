#!/bin/bash
tmux send-keys -t replay 'conda activate RL' Enter 'REPLAY_IP="202.38.64.174" N_ACTORS=8 REGISTERACTORPORT="8079" SENDBATCHPRIORIPORT="8080" UPDATEPRIORIPORT="8081" SAMPLEDATAPORT="8082" python replay.py' Enter
