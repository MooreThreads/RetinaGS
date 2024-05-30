import os  
# 常见列表
# SCENE_NAME = "pointcloud_sample_rate=ALL"
# scene_list=["pointcloud_sample_rate=2", "pointcloud_sample_rate=4", "pointcloud_sample_rate=8", "pointcloud_sample_rate=16", "pointcloud_sample_rate=32", 
            #   "pointcloud_sample_rate=128", "pointcloud_sample_rate=512", "pointcloud_sample_rate=5120"]
# SCENE_NAME= "scene=ALL"
# scene_list=["scene=bicycle", "scene=stump", "scene=treehill", "scene=flowers", "scene=counter", "scene=kitchen", "scene=bonsai"]
# SCENE_NAME= "DGT=0002"
# scene_list=["DGT=00015", "DGT=00010", "DGT=00007", "DGT=00005", "DGT=00003", "DGT=00001"]
# 场景列表  
TEMPLATE_BASH_NAME = 'train_matrix_city_2000_default_PM_DGT=0002'
REPLACE_NAEM = 'DGT=0002'
scene_list=["DGT=00015", "DGT=00010", "DGT=00007", "DGT=00005", "DGT=00003", "DGT=00001"]
SCENE_NAME = 'DGT=ALL'
# 路径
TASK_FOLDER  = '/root/Nerf/Code/DenseGaussian/farm/tasks'
TASK_PENDING_FOLDER = '/root/Nerf/Code/DenseGaussian/farm/tasks_pending'
template_bash_path = os.path.join('/root/Nerf/Code/DenseGaussian/farm/tmplate_task',TEMPLATE_BASH_NAME+'.sh')
    
# 用于记录所有bash文件名的文本文件  
bash_files_log = os.path.join(TASK_PENDING_FOLDER, TEMPLATE_BASH_NAME.replace(REPLACE_NAEM, SCENE_NAME)+'.txt')  
  
# 读取模板bash脚本  
with open(template_bash_path, 'r') as template_file:  
    template_content = template_file.read()  
  
# 遍历场景列表生成bash脚本  
with open(bash_files_log, 'w') as log_file:  
    for scene in scene_list:  
        # 替换场景变量  
        bash_content = template_content.replace(REPLACE_NAEM, scene)  
  
        # 生成新的bash脚本文件名  
        bash_filename = TEMPLATE_BASH_NAME.replace(REPLACE_NAEM, scene) 
        bash_path = os.path.join(TASK_FOLDER, bash_filename+'.sh')  
  
        # 写入新的bash脚本  
        with open(bash_path, 'w') as bash_file:  
            bash_file.write(bash_content)  
  
        # 记录bash文件名到日志文件中  
        log_file.write(bash_filename + '\n')  
  
print(f"Bash scripts have been generated in {TASK_FOLDER}")  
print(f"Log of bash file names has been saved to {bash_files_log}")