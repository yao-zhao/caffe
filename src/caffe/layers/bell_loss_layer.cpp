#include <vector>

#include "caffe/layers/bell_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BellLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.bell_loss_param().power();
  threshold_ = this->layer_param_.bell_loss_param().threshold();
  // CHECK_GE(power_, 0) << "power has to be greater than zeros";
  // CHECK_EQ(power_%2, 0) << "power has to be even number";
  CHECK_GT(threshold_, 0) << "threshold has to be a positive number";
}

template <typename Dtype>
void BellLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BellLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype* tmp_data = diff_.mutable_cpu_diff();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // hack: save the bottom - label in tmp_data
  caffe_sub(count, bottom_data, label_data, tmp_data);
  // calculate normalized diff,
  // temporarily save the normalizer to diff_data first
  caffe_abs(count, label_data, diff_data);
  caffe_cpu_scale(count, Dtype(1)/threshold_, diff_data, diff_data);
  caffe_powx(count, diff_data, power_, diff_data);
  caffe_add_scalar(count, Dtype(1), diff_data);
  caffe_div(count, tmp_data, diff_data, diff_data);
  // calculate loss
  Dtype dot = caffe_cpu_dot(count, diff_data, tmp_data);
  Dtype loss = dot / bottom[0]->count() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BellLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = bottom[0]->count();
    Dtype alpha = top[0]->cpu_diff()[0]/Dtype(count);
    caffe_cpu_scale(count, alpha, diff_.cpu_data(),
      bottom[0]->mutable_cpu_diff());
  }
  LOG_IF(INFO, propagate_down[1]) << "can not propagate down to the label data";
}

#ifdef CPU_ONLY
STUB_GPU(BellLossLayer);
#endif

INSTANTIATE_CLASS(BellLossLayer);
REGISTER_LAYER_CLASS(BellLoss);

}  // namespace caffe
