OUTPUT_DIR="dist"


for python_path in $(find /opt/python/ | grep "cp3[6|7|8|9]")
do
  $python_path/bin/pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  $python_path/bin/python setup.py bdist_wheel -d $OUTPUT_DIR
done

for wheel_file in ./$OUTPUT_DIR/*
do
  mv $wheel_file "${wheel_file/linux/manylinux2014}"
done

