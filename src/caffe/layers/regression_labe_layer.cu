#include <vector>

#include "caffe/layers/regression_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegressionLabelLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data_0 = top[0]->mutable_gpu_data();
  Dtype* top_data_1 = top[1]->mutable_gpu_data();

  caffe_gpu_set(count, lb_, top_data_0);
  caffe_gpu_sub(count, bottom_data, top_data_0, top_data_0);

  caffe_gpu_set(count, ub_, top_data_1);
  caffe_gpu_sub(count, top_data_1, bottom_data, top_data_1);
}

template <typename Dtype>
void RegressionLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(count, Dtype(0), bottom_diff);
  if (propagate_down[0]) {
    const Dtype* top_diff_0 = top[0]->gpu_diff();
    caffe_gpu_add(count, bottom_diff, top_diff_0, bottom_diff);
  }
  if (propagate_down[1]) {
    const Dtype* top_diff_1 = top[1]->gpu_diff();
    caffe_gpu_sub(count, bottom_diff, top_diff_1, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegressionLabelLayer);

}  // namespace caffe
