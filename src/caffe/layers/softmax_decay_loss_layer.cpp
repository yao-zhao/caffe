#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_decay_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom, top);
  method_ = this->layer_param_->softmax_decay_loss_param()->method();
  rate_ = this->layer_param_->softmax_decay_loss_param()->rate();
  softmax_dim_ = bottom[0]->shape(softmax_axis_);
  CHECK_NE(has_ignore_label_, true) << "doesn't alow ignore label";
  CHECK_NE(weight_by_label_freqs_, true) << "doesn't alow label frequency"
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::Reshape(bottom, top);
  weights_.ReshapeLike(*bottom[0]);
  label_idx_.ReshapeLike(*bottom[0]);
  vector<int> dims_mid_inner(1, bottom[0]->count(softmax_axis_));
  mid_inner_multiplier_.Reshape(dims_mid_inner);
  caffe_set(mid_inner_multiplier_.count(), Dtype(1),
      mid_inner_multiplier_.mutable_cpu_data());
  Dtype* label_idx_data = label_idx_.mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < softmax_dim_; ++j) {
      caffe_set(inner_num_, Dtype(j), label_idx_data);
      label_idx_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  Dtype* prob_data = prob_.cpu_data();
  const Dtype* label_idx_data = label_idx_.cpu_data();
  Dtype* weight_data = weights_.mutable_cpu_data();
  const int count = prob_.count();
  // expand label data to channel dimension
  Dtype* weight_iter = weights_.mutable_cpu_data();
  const Dtype* label_iter = bottom[1]->cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < softmax_dim_; ++j) {
      caffe_copy(inner_num_, label_iter, weight_iter);
      weight_iter += inner_num_;
      label_iter += inner_num_;
    }
  }
  // calculate weight
  switch (method_) {
    case SoftmaxWithDecayLossParameter_Decay_GAUSSIAN:
    weight_data = caffe_sub(count, label_idx_data, weight_data, weight_data);
    weight_data = caffe_sqr(count, weight_data, weight_data);
    weight_data = caffe_cpu_scale(count, Dtype(-1), weight_data);
    weight_data = caffe_exp(count, weight_data);
    break;
    case SoftmaxWithDecayLossParameter_Decay_POWER:
    NOT_IMPLEMENTED;
    break;
    default:
    LOG(FATAL) << "Unknown decay method: "
        << SoftmaxWithDecayLossParameter_Decay_Name(method_);
  }
  // weight the prob

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        for (int l = 0; l < dim; ++l) {
          const int idx = i * dim + label_value * inner_num + j;
          weight_data[idx] = weight;
        }
          const int idx = i * dim + label_value * inner_num_ + j;
          ++count;
      }
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithDecayLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithDecayLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithDecayLoss);

}  // namespace caffe
