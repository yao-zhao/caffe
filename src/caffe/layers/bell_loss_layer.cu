#include <vector>

#include "caffe/layers/bell_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BellLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype* tmp_data = diff_.mutable_gpu_diff();
  const Dtype* label_data = bottom[1]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // hack: save the bottom - label in tmp_data
  caffe_gpu_sub(count, bottom_data, label_data, tmp_data);
  // calculate normalized diff,
  // temporarily save the normalizer to diff_data first
  caffe_gpu_abs(count, label_data, diff_data);
  caffe_gpu_scale(count, Dtype(1)/threshold_, diff_data, diff_data);
  caffe_gpu_powx(count, diff_data, power_, diff_data);
  caffe_gpu_add_scalar(count, Dtype(1), diff_data);
  caffe_gpu_div(count, tmp_data, diff_data, diff_data);
  // calculate loss
  Dtype dot;
  caffe_gpu_dot(count, diff_data, tmp_data, &dot);
  Dtype loss = dot / bottom[0]->count() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BellLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = bottom[0]->count();
    Dtype alpha = top[0]->cpu_diff()[0]/Dtype(count);
    caffe_gpu_scale(count, alpha, diff_.gpu_data(),
      bottom[0]->mutable_gpu_diff());
  }
  LOG_IF(INFO, propagate_down[1]) << "can not propagate down to the label data";
}

INSTANTIATE_LAYER_GPU_FUNCS(BellLossLayer);

}  // namespace caffe
