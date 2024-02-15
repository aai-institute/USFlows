
# Installation instructions for marabou
* Checkout commit 5fe8eab67884516a482a600445c870eac530045b
* Follow the install instructions on the marabou page but note:
  * During install with Make/CMake, OpenBlas may not get install correctly. In that case the install process fails and an expressive error is shown. 
  * To fix that, you need to trigger the install of the already downloaded openblas in the marabou directory manually, providing information about your CPU architecture. Again, do this only if cmake fails in the first round using the command:
    * `make TARGET=NEHALEM` in the directory Marabou/tools/OpenBLAS-0.3.19
  * Then retrigger cmake as before - this should run now.
* Then, you need to apply the patch in this directory to Marabou to fix issue [751](https://github.com/NeuralNetworkVerification/Marabou/issues/751)

