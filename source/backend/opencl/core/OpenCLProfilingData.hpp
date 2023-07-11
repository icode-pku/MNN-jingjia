
#ifndef OpenCLProfilingData_hpp
#define OpenCLProfilingData_hpp

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "MNN/MNNDefine.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

template <typename T> std::string VectorToString(std::vector<T> val) {
  if (val.empty())
    return "";

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < val.size(); ++i) {
    stream << val[i];
    if (i != val.size() - 1)
      stream << ",";
  }
  stream << "]";
  return stream.str();
}

std::string DoubleToString(double val);

std::string DoubleToStringFilter(double val);

std::string DimsToString(std::vector<int> dims);

std::vector<std::pair<std::string, std::vector<float>>> 
SortMapByValue(std::map<std::string, std::vector<float>> &map);

std::ostream &FormatRow(std::ostream &stream, int width);

std::string Table(const std::string &title,
                  const std::vector<std::string> &header,
                  const std::vector<std::vector<std::string>> &data);

struct ProfilingData {
  ProfilingData() {}
  ProfilingData(std::string layer, std::string op, double tmp_flops)
      : layer_name(layer), op_name(op), flops(tmp_flops) {}

  ~ProfilingData() {}
  /**layer name*/
  std::string layer_name = "";
  /**op type*/
  std::string op_name = "";
  /**kernel time*/
  double enqueue_time = 0;
  /**submit time*/
  double submit_time = 0;
  /**kernel time*/
  double kernel_time = 0;

  double flops = 0;
  double bandwidth = 0;

  std::vector<int> input_dims = {};
  std::vector<int> output_dims = {};
  std::vector<int> kernel_shape = {};
  std::vector<int> stride_shape = {};
  std::vector<int> pad_shape = {};
  std::vector<int> dilation_shape = {};
  int group = 0;

  int count = 1;

  cl::Event event;
  double event_queued;
  double event_submit;
  double event_start;
  double event_end;
  std::vector<uint32_t> global_worksize = {};
  std::vector<uint32_t> local_worksize = {};

  void Add(ProfilingData *data) {
    if (!data || !IsSameID(data)) {
      return;
    }

    kernel_time += data->kernel_time;
    count += data->count;

    if (input_dims.size() <= 0) {
      input_dims = data->input_dims;
    }

    if (output_dims.size() <= 0) {
      output_dims = data->output_dims;
    }

    if (kernel_shape.size() <= 0) {
      kernel_shape = data->kernel_shape;
    }

    if (stride_shape.size() <= 0) {
      stride_shape = data->stride_shape;
    }

    if (pad_shape.size() <= 0) {
      pad_shape = data->pad_shape;
    }

    if (dilation_shape.size() <= 0) {
      dilation_shape = data->dilation_shape;
    }

    if (group <= 0) {
      group = data->group;
    }
  }
  bool IsSameID(ProfilingData *data) {
    return data && op_name == data->op_name && layer_name == data->layer_name;
  }
};

void GetProfilingTime(ProfilingData *p);

class ProfileResult {
public:
  ~ProfileResult() {}

  // @brief reset for profile again
  void Reset() { profiling_data_.clear(); }

  void AddProfilingData(std::string layer, std::string op, double tmp_flops) {
    auto iter = profiling_data_.find(layer);
    if (iter != profiling_data_.end()) {
      current_data_ = iter->second.get();
    } else {
      std::shared_ptr<ProfilingData> pdata(
          new ProfilingData(layer, op, tmp_flops));
      profiling_data_.insert({pdata->layer_name, pdata});
      current_data_ = pdata.get();
    }
  }

  // @brief add profiling data of each layer
  void AddProfilingData(std::shared_ptr<ProfilingData> pdata) {
    auto iter = profiling_data_.find(pdata->layer_name);
    if (iter != profiling_data_.end()) {
      iter->second->Add(pdata.get());
      current_data_ = iter->second.get();
    } else {
      profiling_data_.insert({pdata->layer_name, pdata});
      current_data_ = pdata.get();
    }
  }

  // @brief add profiling result
  void AddProfileResult(std::shared_ptr<ProfileResult> result) {
    auto result_profiling_data = result->GetData();
    for (auto pf_data : result_profiling_data) {
      AddProfilingData(pf_data.second);
    }
  }

  // @brief get profiling data
  std::map<std::string, std::shared_ptr<ProfilingData>> GetData() {
    return profiling_data_;
  }

  // @brief This function shows the detailed timing for each layer in the model.
  std::string GetProfilingDataInfo() {
    // show the time cost of each layer
    std::string title = "Profiling Data";
    MNN_PRINT("%s\n", title.c_str());
    const std::vector<std::string> header = {
        "name",        "Op Type",      "enqueue(ms)",  "submit(ms)",
        "kernel(ms)",  "flops(ms)",    "FLOPS(ms)",    "BW(GB/s)",
        "Input(NCHW)", "Output(NCHW)", "Filter(OIHW)", "Stride",
        "Pad",         "Dilation",     "GWS(0,1,2)",   "LWS(0,1,2)"};

    std::vector<std::vector<std::string>> data;

    double kernel_time_sum = 0;
    for (auto item : profiling_data_) {
      ProfilingData *p = dynamic_cast<ProfilingData *>(item.second.get());
      if (nullptr == p) {
        printf("ProfilingData is nil\n");
        return "";
      }
      // GetProfiling
      // GetProfilingTime(p);
    }

    for (auto item : profiling_data_) {
      ProfilingData *p = dynamic_cast<ProfilingData *>(item.second.get());
      if (nullptr == p) {
        printf("ProfilingData is nil\n");
        return "";
      }
      std::vector<std::string> tuples = {};
      tuples.reserve(32);

      tuples.push_back(p->layer_name);
      tuples.push_back(p->op_name);
      tuples.push_back(DoubleToString(p->enqueue_time));
      tuples.push_back(DoubleToString(p->submit_time));
      tuples.push_back(DoubleToString(p->kernel_time));
      tuples.push_back(DoubleToString(p->flops));
      tuples.push_back(DoubleToStringFilter(p->flops / p->kernel_time));
      tuples.push_back(DoubleToStringFilter(p->bandwidth / p->kernel_time));
      tuples.push_back(VectorToString(p->input_dims));
      tuples.push_back(VectorToString(p->output_dims));
      tuples.push_back(VectorToString(p->kernel_shape));
      tuples.push_back(VectorToString(p->stride_shape));
      tuples.push_back(VectorToString(p->pad_shape));
      tuples.push_back(VectorToString(p->dilation_shape));
      tuples.push_back(VectorToString(p->global_worksize));
      tuples.push_back(VectorToString(p->local_worksize));

      data.emplace_back(tuples);
      kernel_time_sum += p->kernel_time;
    }

    std::string detailed_string = Table(title, header, data);

    // std::string summary_string = GetProfilingDataSummary(false);
    std::string summary_string = "";

    std::ostringstream ostr;
    ostr << "kernel runtime total: " << kernel_time_sum << " ms\n\n";

    return detailed_string + summary_string + ostr.str();
  }

  ProfilingData *GetCurrentProfilingData() { 
    if (current_data_ == nullptr) {
      MNN_ERROR("current_data_ is nullptr\n");
      return nullptr;
    }
    return current_data_; 
  }

protected:
  /*
   * This function shows an overview of the timings in the model.
   * the timing is grouped by the type of layer.
   */
  std::string GetProfilingDataSummary(bool do_average) {
    // show the time cost of each type layer
    std::string title_summary = "Summary";
    const std::vector<std::string> header_summary = {
        "Op Type", "Total Kernel Time(ms)", "Percent (%)"};

    double kernel_time_sum = 0;
    std::map<std::string, std::vector<float>> summary_map;
    for (auto iter : profiling_data_) {
      auto p = dynamic_cast<ProfilingData *>(iter.second.get());
      if (do_average)
        kernel_time_sum += p->kernel_time / p->count;
      else
        kernel_time_sum += p->kernel_time;
      if (summary_map.find(p->op_name) == summary_map.end()) {
        std::vector<float> p_data;
        p_data.push_back(0.0f);
        summary_map[p->op_name] = p_data;
      }
    }
    for (auto iter : profiling_data_) {
      auto p = dynamic_cast<ProfilingData *>(iter.second.get());
      if (summary_map.find(p->op_name) != summary_map.end()) {
        if (do_average)
          summary_map[p->op_name][0] += p->kernel_time / p->count;
        else
          summary_map[p->op_name][0] += p->kernel_time;
      }
    }
    std::vector<std::pair<std::string, std::vector<float>>> summary_pair(summary_map.begin(),
                                                                   summary_map.end());
    // auto summary_pair = SortMapByValue(summary_map);
    std::vector<std::vector<std::string>> data_summary;
    for (auto s : summary_pair) {
      std::vector<std::string> tuples;
      tuples.reserve(4);

      tuples.push_back(s.first);
      tuples.push_back(DoubleToString(s.second[0]));
      tuples.push_back(DoubleToString(s.second[0] / kernel_time_sum * 100));

      data_summary.emplace_back(tuples);
    }
    std::string show_string_summary =
        Table(title_summary, header_summary, data_summary);
    return show_string_summary;
  }

protected:
  ProfilingData *current_data_ = nullptr;
  std::map<std::string, std::shared_ptr<ProfilingData>> profiling_data_ = {};
};

} // namespace OpenCL
} // namespace MNN

#endif /* OpenCLProfilingData_hpp */
