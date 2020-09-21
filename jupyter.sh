#!/bin/bash


tmux new -s jupyter-notebook -d
tmux send-keys "source activate pytorch1.5.0" C-m
tmux send-keys "jupyter notebook" C-m
