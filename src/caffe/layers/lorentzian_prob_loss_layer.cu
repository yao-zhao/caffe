#include <vector>

#include "caffe/layers/lorentzian_prob_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LorentzianProbLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // hack: save tmp diff for tmp variables, also use diff_diff for save
  // diff_data saves (mean - label)
  // diff_diff saves var
  Dtype* tmp_data = tmp_.mutable_gpu_data();
  Dtype* tmp_data2 = tmp_.mutable_gpu_diff();
  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype* diff_diff = diff_.mutable_gpu_diff();
  const Dtype* mean_data = bottom[0]->gpu_data();
  const Dtype* var_data = bottom[1]->gpu_data();
  const Dtype* label_data = bottom[2]->gpu_data();
  caffe_gpu_sub(count, mean_data, label_data, diff_data);
  caffe_gpu_mul(count, diff_data, diff_data, tmp_data);
  caffe_copy(count, var_data, diff_diff);
  caffe_gpu_add_scalar(count, eps_, diff_diff);
  caffe_gpu_div(count, tmp_data, diff_diff, tmp_data);
  caffe_gpu_add_scalar(count, Dtype(1), tmp_data);
  caffe_gpu_log(count, tmp_data, tmp_data);
  caffe_gpu_log(count, diff_diff, tmp_data2);
  caffe_gpu_scale(count, Dtype(0.5), tmp_data2, tmp_data2);
  caffe_gpu_add(count, tmp_data, tmp_data2, tmp_data);
  // calculate loss
  Dtype loss;
  caffe_gpu_dot(count, tmp_data, sumvec_.gpu_data(), &loss);
  loss /= Dtype(count);
  top[0]->mutable_cpu_data()[0] = loss+Dtype(1.14472988585);
}

template <typename Dtype>
void LorentzianProbLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1]) {
    int count = bottom[0]->count();
    const Dtype alpha_0 = top[0]->cpu_diff()[0]/Dtype(count)*2;
    const Dtype alpha_1 = top[0]->cpu_diff()[0]/Dtype(count)/2;
    Dtype* tmp_data = tmp_.mutable_gpu_data();
    Dtype* tmp_data2 = tmp_.mutable_gpu_diff();
    Dtype* mean_diff = bottom[0]->mutable_gpu_diff();
    Dtype* var_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* diff_data = diff_.mutable_gpu_data();
    const Dtype* diff_diff = diff_.mutable_gpu_diff();
    caffe_gpu_mul(count, diff_data, diff_data, tmp_data2);
    caffe_gpu_add(count, tmp_data2, diff_diff, tmp_data);
    caffe_gpu_div(count, diff_data, tmp_data, tmp_data);
    if (propagate_down[0]) {
      caffe_gpu_scale(count, alpha_0, tmp_data, mean_diff);
    }
    if (propagate_down[1]) {
      caffe_gpu_mul(count, diff_data, tmp_data, tmp_data);
      caffe_gpu_sub(count, sumvec_.gpu_data(), tmp_data, tmp_data);
      caffe_gpu_div(count, tmp_data, diff_diff, tmp_data);
      caffe_gpu_scale(count, alpha_1, tmp_data, var_diff);
    }
  }
  LOG_IF(INFO, propagate_down[2]) << "can not propagate down to the label data";
}

INSTANTIATE_LAYER_GPU_FUNCS(LorentzianProbLossLayer);

}  // namespace caffe
