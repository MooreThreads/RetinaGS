import os, sys
import sqlite3
from argparse import ArgumentParser, Namespace

def save_lines(lines, path):
    with open(os.path.join(path), 'w') as f:
        f.writelines(lines)
        
def change_image_id(db_path, image_txt):
    
    # read db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    select_str = "SELECT image_id, name FROM images"
    cursor.execute(select_str)
    all_images = cursor.fetchall()
    image_dict = {}
    for image in all_images:
        image_dict[image[1]] = image[0]
        
    print('find {} images in db'.format(len(all_images)))
    cursor.close()
    conn.close()
    
    # change image id
    lines = []        
    
    with open(image_txt, "r") as fid:
        while True:
            line = fid.readline()
            if not line:  # 如果读取到了文件末尾，则退出循环
                break
            line = line.strip()  # 去除行首行尾的空白字符
            if len(line):  # 如果行不为空
                elems = line.split()  # 将行按空格分割成多个元素
                image_name = elems[9]
                elems[0] = str(image_dict[image_name])                
                lines.append(' '.join(elems)+'\n')        
                lines.append('\n')        
                
    save_lines(lines, image_txt)            
    

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--db_path', type=str)
    parser.add_argument('--image_txt', type=str)

    args = parser.parse_args(sys.argv[1:])
    db_path, image_txt = args.db_path, args.image_txt

    # db_path = '/root/Nerf/Data/MatrixCity/block_A_unit-1m_choice-64/gt_pose/database.db'
    # image_txt = '/root/Nerf/Data/MatrixCity/block_A_unit-1m_choice-64/gt_pose/sparse/real/images.txt'
    change_image_id(db_path, image_txt)