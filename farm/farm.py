import subprocess  
import time  
import os  
from datetime import datetime 
import shutil

# 使用subprocess执行命令的函数  
def execute_command(command, cwd=None, env=None):  
    process = subprocess.Popen(command, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
    stdout, stderr = process.communicate()  
    if process.returncode != 0:  
        print(f"Error executing command: {stderr.decode()}")  
    return process  
  
# 检查GPU显存并启动任务的函数  
# 阈值（以MB为单位）  
THRESHOLD = 100
# GPU检查间隔（秒）  
CHECK_INTERVAL = 10  
# 启动任务等待，防止多个任务占同一显卡，导致任务启动失败
WATTING_INTERVAL = 60
# 任务文件路径  
TASKS_TO_RUN = '/root/Nerf/Code/DenseGaussian/farm/tasks_to_run.txt'  
TASK_FOLDER  = '/root/Nerf/Code/DenseGaussian/farm/tasks'
TASKS_STARTED = '/root/Nerf/Code/DenseGaussian/farm/task_started'  
MACHINE_ORDER = 'GPU-010'
if not os.path.exists(TASKS_STARTED):
    os.makedirs(TASKS_STARTED, exist_ok=True)
    
def check_and_run_tasks():  
    # 获取所有GPU的显存使用情况  
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']).decode()  
    mem_used_list = [int(line.split(',')[0]) for line in output.strip().split('\n')]  
  
    # 遍历GPU并检查显存使用情况  
    for i, mem_used in enumerate(mem_used_list):  
        if mem_used < THRESHOLD:  # 转换为字节  
            # 读取待执行的任务列表  
            with open(TASKS_TO_RUN, 'r') as f:  
                tasks = f.readlines()  
            if tasks:  
                # 如果有任务，则启动第一个任务并更新任务列表  
                task = tasks[0].strip()  
                # 获取当前时间
                now = datetime.now().strftime('%m%d%H%M')  
                # 使用
                output_path = os.path.join(TASKS_STARTED, f"{task}_{now}_{MACHINE_ORDER}_CUDA-{str(i)}_log.txt")  
                task_path = os.path.join(TASK_FOLDER, task+'.sh')          
                # 设置环境变量  
                env = os.environ.copy()  
                env['CUDA_VISIBLE_DEVICES'] = str(i)          
                # 使用subprocess.Popen  
                command = f'bash {task_path} > {output_path} 2>&1 &'  
                cwd = '/root/Nerf/Code/DenseGaussian'  
                process = execute_command(command, cwd=cwd, env=env)                  
                print(f"{now} GPU {i}: Task '{task}' started.")  
                # print(f"PID of Task '{task}' is: {process.pid}") 
                # 更新待执行的任务列表的任务列表  
                with open(TASKS_TO_RUN, 'w') as f:  
                    f.writelines(tasks[1:])             
                # 将执行的代码也放到task_started里面
                task_started_path=os.path.join(TASKS_STARTED, f"{task}_{now}.sh")
                try:
                    shutil.move(task_path, task_started_path)
                except FileNotFoundError:
                    print("happend to process the same file with other node, just pass it")
                    continue
                time.sleep(WATTING_INTERVAL)            
  
# 主循环，持续检查并启动任务  
while True:  
    check_and_run_tasks()  
    time.sleep(CHECK_INTERVAL)