#!/usr/bin/env bash


DIRS='logs' #텐서보드 로그가 저장되는 폴더명
#LIST='ls -d ${DIRS}/*'
LOGS=''

cd $DIRS

for i in $(ls -d *);do
    if [ "${LOGS}" = '' ] ; then
        LOGS=$i:${DIRS}/$i
    else
        LOGS=$i:${DIRS}/$i,${LOGS}
    fi
done

cd .. # 원래 폴더로 돌아가기

tensorboard --logdir=$LOGS