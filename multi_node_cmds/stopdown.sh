#!/usr/bin/bash
# stopdown.sh，for stop multiple nodes training
 
NODE_LIST=("ip" "ip") 
TIME=$(date "+%Y%m%d-%H%M%S")
MASTER_ADDR=${NODE_LIST[0]}
 

for(( i=0;i<${#NODE_LIST[@]};i++)) do
ssh-copy-id -p 30022 root@${NODE_LIST[i]}
ssh -p 30022 root@${NODE_LIST[i]} echo Rank-${i} Node-${NODE_LIST[i]} is ready!
done;
 

for(( i=0;i<${#NODE_LIST[@]};i++)) do
ssh -p 30022 root@${NODE_LIST[i]} "ps aux|grep torchrun|awk 'NR==1'|awk '{print \$2}'|xargs kill -9"
ssh -p 30022 root@${NODE_LIST[i]} "ps aux|grep main|awk '{print \$2}'|xargs kill -9"
done;
