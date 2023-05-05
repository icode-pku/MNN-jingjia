git config --global user.name "Always"

git config --global user.email "zhouhaoyitian@gmail.com"

ssh-keygen -C "zhouhaoyitian@gmail.com" -t rsa

git config --global user.name "zhouhao03"

git config --global user.email "zhouhaoyitian0318@outlook.com"

ssh-keygen -C "zhouhaoyitian0318@outlook.com" -t rsa

MNN_SUPPORT_BF16 //jjw不支持

# max_debug
cmake  -DCMAKE_BUILD_TYPE=Debug -DMNN_USE_SYSTEM_LIB=ON  -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_DEBUG_MEMORY=ON -DMNN_DEBUG_TENSOR_SIZE=ON -DMNN_GPU_TRACE=ON -DMNN_OPENCL=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=ON -DMNN_OPENCL_PROFILE=ON -DTFMODEL_OPTIMIZE=ON -DMNN_TRAIN_DEBUG=ON -DMNN_BUILD_OPENCV=ON -DMNN_USE_OPENCV=ON -DMNN_IMGPROC_COLOR=ON -DMNN_IMGPROC_GEOMETRIC=ON -DMNN_IMGCODECS=ON -DMNN_CALIB3D=OFF -DMNN_OPENCV_TEST=OFF ../..
# max_release
cmake  -DCMAKE_BUILD_TYPE=Release -DMNN_USE_SYSTEM_LIB=ON  -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_DEBUG_MEMORY=ON -DMNN_DEBUG_TENSOR_SIZE=ON -DMNN_GPU_TRACE=ON -DMNN_OPENCL=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=ON -DMNN_OPENCL_PROFILE=ON -DTFMODEL_OPTIMIZE=ON -DMNN_TRAIN_DEBUG=ON -DMNN_BUILD_OPENCV=ON -DMNN_USE_OPENCV=ON -DMNN_IMGPROC_COLOR=ON -DMNN_IMGPROC_GEOMETRIC=ON -DMNN_IMGCODECS=ON -DMNN_CALIB3D=OFF -DMNN_OPENCV_TEST=OFF ../..

# min_debug
cmake -DCMAKE_BUILD_TYPE=Debug -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_DEMO=ON -DMNN_OPENCL=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ../..
# min_release
cmake -DCMAKE_BUILD_TYPE=Release -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_DEMO=ON -DMNN_OPENCL=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ../..


# segment
## convert 
chmod 777 MNNConvert
./MNNConvert -f TFLITE --modelFile ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.tflite --MNNModel ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn --bizCode biz
./MNNConvert -f MNN --modelFile ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn --JsonFile ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn.json
./MNNConvert -f JSON --modelFile ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn.json --MNNModel ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn.json.mnn

## demo
chmod 777 segment.out
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/mwv207-fixed/
./segment.out ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn ../../resource/zhouhao03/segment/segment_input.jpg ../../resource/zhouhao03/segment/max_segment_output.jpg 
./segment.out ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn ../../resource/zhouhao03/segment/segment_input.jpg ../../resource/zhouhao03/segment/cpu_min_segment_output.jpg 
./segment.out ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn ../../resource/zhouhao03/segment/segment_input.jpg ../../resource/zhouhao03/segment/ocl_min_segment_output.jpg 

## tool
./GetMNNInfo ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn

## backend
./backendTest.out ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn 3

## basic
./MNNV2Basic.out model [runLoops runMask forwardType numberThread inputSize precision]
./MNNV2Basic.out ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn 10 0 3 4

python ../../tools/script/testMNNFromTflite.py ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.tflite
./ModuleBasic ../../resource/zhouhao03/segment/deeplabv3_257_mv_gpu.mnn 10 0 0 4

# yolo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/mwv207-fixed/
./timeProfile.out /home/zhouhao/github/MNN/zhdata/yolov5_jjw/yolov5s.mnn
./yolov5s_jjw.out /home/zhouhao/github/MNN/zhdata/yolov5_jjw/000070.jpg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/mwv207-fixed/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/mwv207-fixed/
export LD_PRELOAD=/opt/mwv207-compute/clPerfTools/lib/libOpenCL.so.1.2 
export CLI_OpenCLFileName=/usr/lib/aarch64-linux-gnu/mwv207-fixed/libOpenCL.so 
export CLI_ChromeCallLogging=1
export CLI_ChromePerformanceTiming=1 
export CLI_ChromePerformanceTimingInStages=1
export CLI_ChromePerformanceTimingPerKernel=1
export CLI_LogToFile=1 
export CLI_DumpDir=/home/zhouhao/github/zhouhao03/MNN-dev/build/clperf/
./yolov5s_jjw.out /home/zhouhao/github/MNN/zhdata/yolov5_jjw/000070.jpg

./timeProfile.out ../../resource/zhouhao03/yolov5_jjw/yolov5s.mnn
./yolov5s_jjw.out ../../resource/zhouhao03/yolov5_jjw/000070.jpg

# common
./getPerformance