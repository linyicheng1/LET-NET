python setup.py build_ext --inplace
mv project_depth_nn_cython_pkg.*.so project_depth_nn_cython_pkg.so
# clean build output
rm project_depth_nn_cython.c
rm project_depth_nn_cython.html
rm -rf build