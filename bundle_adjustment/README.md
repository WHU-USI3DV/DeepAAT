# Python Bundle Adjustment
These instructions are based on and almost identical to the ones at [https://github.com/drormoran/gasfm/tree/main/bundle_adjustment/README.md](https://github.com/drormoran/gasfm/tree/main/bundle_adjustment/README.md).


## Conda envorinment
Use the <a href="https://github.com/WHU-USI3DV/DeepAAT/blob/main/environment.yml">gasfm</a> environment.
```
conda activate deepaat
export PYBIND11_PYTHON_VERSION="3.8"
export PYTHON_VERSION="3.8"
```

## Directory structure
After this set up, the directory structure be:
```
DeepAAT
├── bundle_adjustment
│   ├── ceres-solver
│   │   ├── ceres-bin
│   │   |   └── lib
│   │   |       └── PyCeres.cpython-39-x86_64-linux-gnu.so
│   │   ├── ceres_python_bindings
│   │   |   └── python_bindings
│   │   |       └── custom_cpp_cost_functions.cpp
│   │   └── CMakeLists.txt
│   └── custom_cpp_cost_functions.cpp
├── code
├── datasets
```
## Set up
1. Clone the <a href="http://ceres-solver.org/installation.html">Ceres-Solver</a> repository to the bundle_adjustment folder and check out version 2.1.0:

```
cd bundle_adjustment
git clone https://ceres-solver.googlesource.com/ceres-solver -b 2.1.0
```


2. Clone the <a href="https://github.com/Edwinem/ceres_python_bindings">ceres_python_bindings</a> package inside the ceres-solver folder:

```
cd ceres-solver
git clone https://github.com/Edwinem/ceres_python_bindings
```


3. Copy the file "custom_cpp_cost_functions.cpp" and replace the file "ceres-solver/ceres_python_bindings/python_bindings/custom_cpp_cost_functions.cpp".
This file contains projective and euclidean custom bundle adjustment functions.

```
cp ../custom_cpp_cost_functions.cpp ceres_python_bindings/python_bindings/custom_cpp_cost_functions.cpp
```

Next, you need to build ceres_python_bindings and ceres-solver and create a shared object file that python can call.
You can either continue with the instructions here or follow the instructions at the <a href="https://github.com/Edwinem/ceres_python_bindings">ceres_python_bindings</a> repository.

1. run:

```
cd ceres_python_bindings
git submodule init
git submodule update
```


1. Make sure that the C++ standard library version used during the build is recent enough, and not hard-coded to C++11 by pybind11. Please check your c++ compiler version and modify it bellow (for example here is c++17):

```
sed -i 's/set(PYBIND11_CPP_STANDARD -std=c++11)/set(PYBIND11_CPP_STANDARD -std=c++17)/g' AddToCeres.cmake
```


5. Add to the end of the file ceres-solver/CMakeLists.txt the line: "include(ceres_python_bindings/AddToCeres.cmake)":

```
cd ..
echo "include(ceres_python_bindings/AddToCeres.cmake)" >> CMakeLists.txt
```


6. Inside ceres-solver folder run:


```
mkdir ceres-bin
cd ceres-bin
cmake ..
make -j8
make test
```

7. If everything worked you should see the following file:

```
bundle_adjustment/ceres-solver/ceres-bin/lib/PyCeres.cpython-39-x86_64-linux-gnu.so
```

8. If you want to use this bundle adjustment implementation for a different project make sure to add the path of the shared object to linux PATH (in the code this is done for you). In the python project this would be for example:

```
import sys
sys.path.append('../bundle_adjustment/ceres-solver/ceres-bin/lib/')
import PyCeres
```

To see the usage of the PyCeres functions go to code/utils/ceres_utils and code/utils/ba_functions.

## Note
If you encounter problems while compiling PyCeres, you can also refer to [GASFM](https://github.com/lucasbrynte/gasfm)'s [bundle adjustment instruction](https://github.com/lucasbrynte/gasfm/blob/main/bundle_adjustment/README.md) or create a new environment for compilation. 
If the environment for compiling PyCeres is not the same as the network environment, then run_ba in conf file can be set to False during training and inference, and after obtaining the predicted results, run run_ba.py separately for BA.