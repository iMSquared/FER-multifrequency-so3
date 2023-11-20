apt-get update
apt-get install libgl1-mesa-glx -y
cd pointnet2_ops_lib
pip install .
cd ..
pip install -r requirements.txt
pip install --upgrade scipy