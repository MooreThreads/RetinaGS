#!/usr/bin/bash
# startup.sh，for launching multiple nodes training
 
# ip list of nodes, first of which would be master node 
NODE_LIST=("ip" "ip") 

# NODE_NUM为节点数量
NODE_NUM=${#NODE_LIST[@]}
# 时间标签
TIME=$(date "+%Y%m%d-%H%M%S")
# master节点为节点列表中的第一个
MASTER_ADDR=${NODE_LIST[0]}
 
# just make every node run the same shell script
RUN_SCRIPT=path/2/start_multi_node.sh
 
# authorise all nodes via ssh
for(( i=0;i<${#NODE_LIST[@]};i++)) do
# copy key to nodes, this step would be automatically skipped if it had been done
ssh-copy-id -p 30022 root@${NODE_LIST[i]} 
ssh -p 30022 root@${NODE_LIST[i]} echo Node_rank-${i} Node-${NODE_LIST[i]} is ready!
done;
 
# launch RUN_SCRIPT
for(( i=0;i<${#NODE_LIST[@]};i++)) do
# 每个节点上面运行的日志文件地址
LOG_FILE=path/2/log/$TIME-${i}.log

echo Node_rank-${i}, Node-${NODE_LIST[i]}, $LOG_FILE
# set paramters loke NODE_RANK、NNODES for RUN_SCRIPT
ssh -p 30022 root@${NODE_LIST[i]} NODE_RANK=${i} NNODES=$NODE_NUM MASTER_ADDR=$MASTER_ADDR nohup bash $RUN_SCRIPT > $LOG_FILE 2>&1 &
done;