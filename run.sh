dataset_path="/media/madhav/3DRecons-Data/Reconstruction-Dataset/openvslam-dataset/aist_living_lab_3/images/"
src_filename="frame100"
src_extension=".jpg"


python3 preprocess.py --img_glob ${dataset_path}${src_filename}${src_extension} --output_dir assets/preprocessed/
python3 inference.py --pth ckpt/resnet50_rnn__st3d.pth --img_glob assets/preprocessed/${src_filename}_aligned_rgb.png --output_dir assets/inferenced --visualize
python3 layout_viewer.py --img assets/preprocessed/${src_filename}_aligned_rgb.png --layout assets/inferenced/${src_filename}_aligned_rgb.json --ignore_ceiling --vis
