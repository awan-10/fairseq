# editable was failing on our clusters
pip install --user .
python setup.py build_ext --inplace
