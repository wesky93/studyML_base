#!/usr/bin/env bash
# 학습 스크립트 자동으로 깃풀 한뒤 학습을 실행시키고 이후 텐서보드를 실행한다.
git pull

python3 agent.py

source tensorboard.sh