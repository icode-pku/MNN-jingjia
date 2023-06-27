#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/opencv.hpp>

#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include "CL/cl2.hpp"

#if 1
inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }
#else
static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}
#endif

// Computes IOU between two bounding boxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) {
  float in = (bb_test & bb_gt).area();
  float un = bb_test.area() + bb_gt.area() - in;

  if (un < DBL_EPSILON)
    return 0;

  return (double)(in / un);
}

namespace yolocv {
typedef struct {
  int width;
  int height;
} YoloSize;
} // namespace yolocv

typedef struct {
  std::string name;
  int stride;
  std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

class BoxInfo {
public:
  int x1, y1, x2, y2, label, id;
  float score;
};

std::vector<BoxInfo> decode_infer(MNN::Tensor &data, int stride,
                                  const yolocv::YoloSize &frame_size,
                                  int net_size, int num_classes,
                                  const std::vector<yolocv::YoloSize> &anchors,
                                  float threshold) {
  std::vector<BoxInfo> result;
  int batchs, channels, height, width, pred_item;
  batchs = data.shape()[0];
  channels = data.shape()[1];
  height = data.shape()[2];
  width = data.shape()[3];
  pred_item = data.shape()[4];

  auto data_ptr = data.host<float>();
  for (int bi = 0; bi < batchs; bi++) {
    auto batch_ptr = data_ptr + bi * (channels * height * width * pred_item);
    for (int ci = 0; ci < channels; ci++) {
      auto channel_ptr = batch_ptr + ci * (height * width * pred_item);
      for (int hi = 0; hi < height; hi++) {
        auto height_ptr = channel_ptr + hi * (width * pred_item);
        for (int wi = 0; wi < width; wi++) {
          auto width_ptr = height_ptr + wi * pred_item;
          auto cls_ptr = width_ptr + 5;

          auto confidence = sigmoid(width_ptr[4]);

          for (int cls_id = 0; cls_id < num_classes; cls_id++) {
            float score = sigmoid(cls_ptr[cls_id]) * confidence;
            if (score > threshold) {
              float cx =
                  (sigmoid(width_ptr[0]) * 2.f - 0.5f + wi) * (float)stride;
              float cy =
                  (sigmoid(width_ptr[1]) * 2.f - 0.5f + hi) * (float)stride;
              float w = pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[ci].width;
              float h =
                  pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[ci].height;

              BoxInfo box;

              box.x1 =
                  std::max(0, std::min(frame_size.width, int((cx - w / 2.f))));
              box.y1 =
                  std::max(0, std::min(frame_size.height, int((cy - h / 2.f))));
              box.x2 =
                  std::max(0, std::min(frame_size.width, int((cx + w / 2.f))));
              box.y2 =
                  std::max(0, std::min(frame_size.height, int((cy + h / 2.f))));
              box.score = score;
              box.label = cls_id;
              result.push_back(box);
            }
          }
        }
      }
    }
  }

  return result;
}

void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float h = std::max(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

using namespace std;
using namespace MNN;

#define MNN_OPEN_TIME_TRACE

void show_shape(std::vector<int> shape) {
  //   std::cout << shape[0] << " " << shape[1] << " " << shape[2] << " " <<
  //   shape[3]
  //             << " " << shape[4] << " " << std::endl;
}

void scale_coords(std::vector<BoxInfo> &boxes, int w_from, int h_from, int w_to,
                  int h_to) {
  float w_ratio = float(w_to) / float(w_from);
  float h_ratio = float(h_to) / float(h_from);

  for (auto &box : boxes) {
    box.x1 *= w_ratio;
    box.x2 *= w_ratio;
    box.y1 *= h_ratio;
    box.y2 *= h_ratio;
  }
  return;
}

cv::Mat draw_box(cv::Mat &cv_mat, std::vector<BoxInfo> &boxes,
                 const std::vector<std::string> &labels) {
  int CNUM = 80;
  cv::RNG rng(0xFFFFFFFF);
  cv::Scalar_<int> randColor[CNUM];
  for (int i = 0; i < CNUM; i++)
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

  for (auto box : boxes) {
    randColor[box.label] = cv::Scalar(0, 0, 255);
    int width = box.x2 - box.x1;
    int height = box.y2 - box.y1;
    int id = box.id;
    cv::Point p = cv::Point(box.x1, box.y1);
    cv::Rect rect = cv::Rect(box.x1, box.y1, width, height);
    cv::rectangle(cv_mat, rect, randColor[box.label]);
    string text = labels[box.label] + ":" + std::to_string(box.score) +
                  " ID:" + std::to_string(id);
    cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
                randColor[box.label]);
  }
  return cv_mat;
}

int main(int argc, const char *argv[]) {
  auto handle = dlopen("/home/zhouhao/github/zhouhao03/MNN-dev/build/min_debug/source/backend/opencl/libMNN_CL.so", RTLD_NOW);
  // auto handle = dlopen("./source/backend/opencl/libMNN_CL.so", RTLD_NOW);
  if (handle == nullptr) {
    MNN_PRINT("dlopen libMNN_CL.so failed\n");
    return 0;
  }
  if (argc < 2) {
    MNN_PRINT("Usage: ./yolov5 input.jpg [forwardType] [precision] \n");
    return 0;
  }

  int isVedio = 0;
#if 1
  std::string model_name = "/home/zhouhao/github/zhouhao03/MNN-dev/resource/"
                           "zhouhao03/yolov5_jjw/yolov5s.mnn";
  int num_classes = 80;
  std::vector<YoloLayerData> yolov5s_layers{
      {"437", 32, {{116, 90}, {156, 198}, {373, 326}}},
      {"417", 16, {{30, 61}, {62, 45}, {59, 119}}},
      {"output", 8, {{10, 13}, {16, 30}, {33, 23}}},
  };
  std::vector<YoloLayerData> &layers = yolov5s_layers;
  std::vector<std::string> labels{
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
#else

  std::string model_name = "/home/zhouhao/github/zhouhao03/MNN-dev/resource/"
                           "zhouhao03/yolov5_jjw/yolov5ss.mnn";
  int num_classes = 5;
  // auto handle = dlopen("libMNN_CL.so", RTLD_NOW);
  std::vector<YoloLayerData> yolov5ss_layers{
      {"415", 32, {{116, 90}, {156, 198}, {373, 326}}},
      {"395", 16, {{30, 61}, {62, 45}, {59, 119}}},
      {"output", 8, {{10, 13}, {16, 30}, {33, 23}}},
  };
  std::vector<YoloLayerData> &layers = yolov5ss_layers;
  std::vector<std::string> labels{"person", "vehicle", "outdoor", "animal",
                                  "accessory"};
#endif

  // auto revertor = std::unique_ptr<Revert>(new Revert(model_name.c_str()));
  // revertor->initialize();
  // auto modelBuffer      = revertor->getBuffer();
  // const auto bufferSize = revertor->getBufferSize();
  // auto net =
  // std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer,
  // bufferSize)); revertor.reset();

  std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_name.c_str()));
  if (nullptr == net) {
    return 0;
  }

  MNN::ScheduleConfig config;
  config.type = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
  if (argc > 2) {
    config.type = (MNNForwardType)::atoi(argv[2]);
  }
  printf("forward type: %d\n", config.type);
  config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_WIDE;
  if (argc > 3) {
    config.mode = atoi(argv[3]);
  }
  printf("forward mode: %d\n", config.mode);

  // 外部建立OpenCL Context
  std::vector<cl::Platform> platforms;
  cl_int res = cl::Platform::get(&platforms);
  if (res != CL_SUCCESS) {
    printf("clGetPlatformIDs error!\n");
    return 0;
  }
  cl::Platform::setDefault(platforms[0]);
  std::vector<cl::Device> gpuDevices;
  res = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
  std::shared_ptr<cl::Device> mFirstGPUDevicePtr = std::make_shared<cl::Device>(gpuDevices[0]);
  const std::string deviceName = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NAME>();
  const std::string deviceVersion = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VERSION>();
  std::shared_ptr<cl::Context> mContext = std::shared_ptr<cl::Context>(new cl::Context(std::vector<cl::Device>({*mFirstGPUDevicePtr}), nullptr, nullptr, nullptr, &res));
  // MNN_CHECK_CL_SUCCESS(res, "context");
  // if (res != CL_SUCCESS) {
  //     mIsCreateError = true;
  //     return;
  // }
  cl_command_queue_properties properties = 0;
  std::shared_ptr<cl::CommandQueue> mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &res);
  // MNN_CHECK_CL_SUCCESS(res, "commandQueue");
  // if (res != CL_SUCCESS) {
  //     mIsCreateError = true;
  //     return;
  // }
  cl_command_queue cl_cq = mCommandQueuePtr.get()->get();

  MNN::BackendConfig backendConfig;
  backendConfig.sharedContext = (void *)cl_cq;
  backendConfig.precision = MNN::BackendConfig::Precision_Low;
  if (argc > 4) {
    backendConfig.precision =
        (MNN::BackendConfig::PrecisionMode)::atoi(argv[4]);
  }
  printf("backendConfig.precision: %d\n", backendConfig.precision);
  config.backendConfig = &backendConfig;
  MNN::Session *session = net->createSession(config);

  int INPUT_SIZE = 640;

  cv::Mat img_src;
  cv::VideoCapture cap;
  cv::Mat image;
  const cv::Size inputSize_ = cv::Size(INPUT_SIZE, INPUT_SIZE);

  std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
  auto nhwc_Tensor =
      MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
  auto nhwc_data = nhwc_Tensor->host<float>();
  auto nhwc_size = nhwc_Tensor->size();

  if (strcmp(argv[1], "0") == 0) {
    cap.open(0);
    isVedio = 1;
  } else {
    cap.open(argv[1]);
  }

  CV_Assert(cap.isOpened());

  if (cap.get(cv::CAP_PROP_FRAME_COUNT) > 1)
    isVedio = 1;

  while (1) {
    cap >> img_src;
    if (img_src.empty()) {
      printf("exit\n");
      // waitKey();
      break;
    }
    cv::Mat image;
    if (inputSize_ != img_src.size()) {
    }

    struct timeval timestart, timeend;
    uint64_t pre_timecost = 0;
    uint64_t infer_timecost = 0;
    uint64_t post_timecost = 0;

    int iter_len = 1;

    for (int i = 0; i < iter_len; ++i) {
      printf("i = %d\n", i);
      gettimeofday(&timestart, NULL);
      cv::resize(img_src, image, cv::Size(INPUT_SIZE, INPUT_SIZE));

      // preprocessing
      image.convertTo(image, CV_32FC3);
      // image = (image * 2 / 255.0f) - 1;
      image = image / 255.0f;

      // wrapping input tensor, convert nhwc to nchw

      std::memcpy(nhwc_data, image.data, nhwc_size);
      gettimeofday(&timeend, NULL);
      pre_timecost += (timeend.tv_sec * 1000000lu + timeend.tv_usec) -
                      (timestart.tv_sec * 1000000lu + timestart.tv_usec);

      // gettimeofday(&timestart, NULL);

      // auto inputTensor = net->getSessionInput(session, nullptr);
      // inputTensor->copyFromHostTensor(nhwc_Tensor);
      //  gettimeofday(&timeend, NULL);
      // timecost = (timeend.tv_sec*1000000lu+timeend.tv_usec) -
      // (timestart.tv_sec*1000000lu+timestart.tv_usec); printf(
      // "copyFromHostTensor timecost = %.4f ms\n", (double)timecost/1000.0f);

      gettimeofday(&timestart, NULL);
      auto inputTensor = net->getSessionInput(session, nullptr);
      inputTensor->copyFromHostTensor(nhwc_Tensor);

      // run network
      net->runSession(session);

      std::string output_tensor_name0 = layers[2].name;
      std::string output_tensor_name1 = layers[1].name;
      std::string output_tensor_name2 = layers[0].name;

      MNN::Tensor *tensor_scores =
          net->getSessionOutput(session, output_tensor_name0.c_str());
      MNN::Tensor *tensor_boxes =
          net->getSessionOutput(session, output_tensor_name1.c_str());
      MNN::Tensor *tensor_anchors =
          net->getSessionOutput(session, output_tensor_name2.c_str());

      MNN::Tensor tensor_scores_host(tensor_scores,
                                     tensor_scores->getDimensionType());
      MNN::Tensor tensor_boxes_host(tensor_boxes,
                                    tensor_boxes->getDimensionType());
      MNN::Tensor tensor_anchors_host(tensor_anchors,
                                      tensor_anchors->getDimensionType());

      tensor_scores->copyToHostTensor(&tensor_scores_host);
      tensor_boxes->copyToHostTensor(&tensor_boxes_host);
      tensor_anchors->copyToHostTensor(&tensor_anchors_host);
      // clFinsh();

      gettimeofday(&timeend, NULL);
      infer_timecost += (timeend.tv_sec * 1000000lu + timeend.tv_usec) -
                        (timestart.tv_sec * 1000000lu + timestart.tv_usec);

      gettimeofday(&timestart, NULL);
      std::vector<BoxInfo> result;
      std::vector<BoxInfo> boxes;

      yolocv::YoloSize yolosize = yolocv::YoloSize{INPUT_SIZE, INPUT_SIZE};

      float threshold = 0.3;
      float nms_threshold = 0.7;

      show_shape(tensor_scores_host.shape());
      show_shape(tensor_boxes_host.shape());
      show_shape(tensor_anchors_host.shape());

      boxes =
          decode_infer(tensor_scores_host, layers[2].stride, yolosize,
                       INPUT_SIZE, num_classes, layers[2].anchors, threshold);
      result.insert(result.begin(), boxes.begin(), boxes.end());

      boxes =
          decode_infer(tensor_boxes_host, layers[1].stride, yolosize,
                       INPUT_SIZE, num_classes, layers[1].anchors, threshold);
      result.insert(result.begin(), boxes.begin(), boxes.end());

      boxes =
          decode_infer(tensor_anchors_host, layers[0].stride, yolosize,
                       INPUT_SIZE, num_classes, layers[0].anchors, threshold);
      result.insert(result.begin(), boxes.begin(), boxes.end());

      nms(result, nms_threshold);

      // std::cout<<result.size()<<std::endl;

      scale_coords(result, INPUT_SIZE, INPUT_SIZE, img_src.cols, img_src.rows);

      gettimeofday(&timeend, NULL);
      post_timecost += (timeend.tv_sec * 1000000lu + timeend.tv_usec) -
                       (timestart.tv_sec * 1000000lu + timestart.tv_usec);

      draw_box(img_src, result, labels);
    }

    printf("preprocess timecost = %.4f ms\n",
           (double)pre_timecost / 1000.0f / iter_len);
    printf("infer timecost = %.4f ms\n",
           (double)infer_timecost / 1000.0f / iter_len);
    printf("postprocess timecost = %.4f ms\n",
           (double)post_timecost / 1000.0f / iter_len);

    if (argc > 4 && atoi(argv[4]) == 1) {
      // cv::imshow("Yolov5", img_src);
      cv::imwrite("/home/zhouhao/github/zhouhao03/MNN-dev/resource/zhouhao03/"
                  "yolov5_jjw/output.jpg",
                  img_src);
      cv::waitKey(1);
    }

    cv::imwrite("/home/zhouhao/github/zhouhao03/MNN-dev/resource/zhouhao03/"
                "yolov5_jjw/output_new.jpg",
                img_src);

    if (isVedio == 0)
      break;
  }
  return 0;
}
