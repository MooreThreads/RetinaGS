DATA_PATH=/jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap/Mill_19/rubble-pixsfm

mkdir $DATA_PATH/gt_pose
colmap feature_extractor \
    --ImageReader.camera_model PINHOLE \
    --database_path "$DATA_PATH/gt_pose/database.db" \
    --image_path "$DATA_PATH/images" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

# 改内参 (optional)
# from https://colmap.github.io/faq.html:
# If your known camera intrinsics have large distortion coefficients, 
# you should now manually copy the parameters from your cameras.txt to the database, 
# such that the matcher can leverage the intrinsics. 
python scripts/colmap_intrinsics.py \
    --db_path "$DATA_PATH/gt_pose/database.db" \
    --camera_txt "$DATA_PATH/sparse/cameras.txt"

# 改image ID，对应到db 
# you might set change camera_id of images in this step 
# if cameras in db are not modified to match images
python scripts/change_image_id.py \
    --db_path "$DATA_PATH/gt_pose/database.db" \
    --image_txt "$DATA_PATH/sparse/images.txt"

colmap exhaustive_matcher \
    --database_path "$DATA_PATH/gt_pose/database.db" \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0,1,2,3

mkdir -p $DATA_PATH/sparse/0

colmap point_triangulator \
    --database_path "$DATA_PATH/gt_pose/database.db" \
    --image_path "$DATA_PATH/images" \
    --input_path "$DATA_PATH/sparse" \
    --output_path "$DATA_PATH/sparse/0" \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0

# Note that the sparse reconstruction step is not necessary in order to compute a dense model from known camera poses. 
# Assuming you computed a sparse model from the known camera poses, you can compute a dense model as follows:
colmap patch_match_stereo \
    --workspace_path "$DATA_PATH/gt_pose/dense" \
    --PatchMatchStereo.gpu_index 7

colmap stereo_fusion \
    --workspace_path "$DATA_PATH/gt_pose/dense" \
    --output_path "$DATA_PATH/gt_pose/dense/fused.ply"

