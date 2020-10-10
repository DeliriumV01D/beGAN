set opencv_release_path=C:\opencv_4_3\build\bin\Release
set opencv_debug_path=C:\opencv_4_3\build\bin\Debug
set mxnet_release=C:\mxnet\build\Release
set mxnet_debug=C:\mxnet\build\Debug
set path=%opencv_release_path%;%opencv_debug_path%;%mxnet_release%;%mxnet_debug%;%path%

set MXNET_CUDNN_AUTOTUNE_DEFAULT=0

start "" "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\devenv.exe" build\beGAN.sln

