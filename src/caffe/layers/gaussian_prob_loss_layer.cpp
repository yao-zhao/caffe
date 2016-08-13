#include <vector>

#include "caffe/layers/gaussian_prob_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void GaussianProbLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  eps_ = this->layer_param_.gaussian_prob_loss_param().eps();
  CHECK_GT(eps_, 0)
  <<"eps has to be a positive number to ensure nonzero variance";
}

template <typename Dtype>
void GaussianProbLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
    << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(), bottom[2]->count())
    << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  tmp_.ReshapeLike(*bottom[0]);
  sumvec_.ReshapeLike(*bottom[0]);
  caffe_set(bottom[0]->count(), Dtype(1), sumvec_.mutable_cpu_data());
}

template <typename Dtype>
void GaussianProbLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // hack: save tmp diff for tmp variables, also use diff_diff for save
  // diff_data saves (mean - label)
  // diff_diff saves var
  Dtype* tmp_data = tmp_.mutable_cpu_data();
  Dtype* tmp_data2 = tmp_.mutable_cpu_diff();
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype* diff_diff = diff_.mutable_cpu_diff();
  const Dtype* mean_data = bottom[0]->cpu_data();
  const Dtype* var_data = bottom[1]->cpu_data();
  const Dtype* label_data = bottom[2]->cpu_data();
  caffe_sub(count, mean_data, label_data, diff_data);
  caffe_sqr(count, diff_data, tmp_data);
  caffe_copy(count, var_data, diff_diff);
  caffe_add_scalar(count, eps_, diff_diff);
  caffe_div(count, tmp_data, diff_diff, tmp_data);
  caffe_log(count, diff_diff, tmp_data2);
  caffe_add(count, tmp_data, tmp_data2, tmp_data);
  // calculate loss
  Dtype loss = caffe_cpu_dot(count, tmp_data, sumvec_.cpu_data());
  loss /= Dtype(count)*Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss+Dtype(0.3990899);
}

template <typename Dtype>
void GaussianProbLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1]) {
    int count = bottom[0]->count();
    Dtype alpha = top[0]->cpu_diff()[0]/Dtype(count);
    Dtype* mean_diff = bottom[0]->mutable_cpu_diff();
    Dtype* var_diff = bottom[1]->mutable_cpu_diff();
    Dtype* diff_data = diff_.mutable_cpu_data();
    Dtype* diff_diff = diff_.mutable_cpu_diff();
    caffe_div(count, diff_data, diff_diff, diff_data);
    if (propagate_down[0]) {
      caffe_cpu_scale(count, alpha, diff_data, mean_diff);
    }
    if (propagate_down[1]) {
      caffe_sqr(count, diff_data, diff_data);
      caffe_inv(count, diff_diff, diff_diff);
      caffe_sub(count, diff_diff, diff_data, var_diff);
      caffe_cpu_scale(count, alpha, var_diff, var_diff);
    }
  }
  LOG_IF(INFO, propagate_down[2])<<"can not propagate down to the label data";
}

#ifdef CPU_ONLY
STUB_GPU(GaussianProbLossLayer);
#endif

INSTANTIATE_CLASS(GaussianProbLossLayer);
REGISTER_LAYER_CLASS(GaussianProbLoss);

}  // namespace caffe
