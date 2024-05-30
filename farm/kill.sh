# 查看具体情况，会给出当时的详细命令，再根据PID kill即可
ps aux|grep render_metric
# 查看后手动kill
kill -9 xxx
# kill所有带render_metric名字的进程
ps aux|grep train_with_dataset.py|awk '{print $2}'|xargs kill -9
