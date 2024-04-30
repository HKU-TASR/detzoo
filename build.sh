echo "Building..."
python setup.py clean --all
python setup.py sdist bdist_wheel

rm -rf bin
mkdir bin
mv build bin
mv detzoo.egg-info bin
mv dist bin

pip install --user bin/dist/detzoo-0.1-py3-none-any.whl --force-reinstall
