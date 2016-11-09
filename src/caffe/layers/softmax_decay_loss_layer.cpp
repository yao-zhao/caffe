#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_decay_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_param.clear_loss_weight();
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  method_ = this->layer_param_.softmax_decay_loss_param().method();
  rate_ = this->layer_param_.softmax_decay_loss_param().rate();
  CHECK_NE(this->layer_param_.loss_param().has_ignore_label(), true)
      << "doesn't alow ignore label";
  CHECK_NE(this->layer_param_.loss_param().weight_by_label_freqs(), true)
      << "doesn't alow label frequency";
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  weights_.ReshapeLike(*bottom[0]);
  softmax_dim_ = bottom[0]->shape(softmax_axis_);
}

template <typename Dtype>
inline Dtype get_weight_gaussian_cpu(Dtype diff_label, Dtype rate) {
  return exp(-(diff_label * diff_label) / (rate * rate));
}

template <typename Dtype>
inline void forward_cpu_kernel(Dtype(*weight_func)(Dtype, Dtype),
    const Dtype* prob_data, const Dtype* label_data, Dtype* weight_data,
    const int outer_num, const int inner_num, const int softmax_dim_,
    const Dtype rate, 
    Dtype* loss) {
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num; j++) {
      const Dtype label_value = label_data[i * inner_num + j];
      Dtype sum_weights = 0;
      Dtype sum_loss = 0;
      for (int l = 0; l < softmax_dim_; ++l) {
        const int idx = (i * softmax_dim_ + l) * inner_num + j;
        const Dtype weight = weight_func(Dtype(l) - label_value, rate);
        sum_weights += weight;
        sum_loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN))) * weight;
        weight_data[idx] = weight;
      }
      if (sum_weights > 0) {
        *loss += sum_loss/sum_weights;
        for (int l = 0; l < softmax_dim_; ++l) {
          const int idx = (i * softmax_dim_ + l) * inner_num + j;
          weight_data[idx] /= sum_weights;
        }
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* weight_data = weights_.mutable_cpu_data();
  Dtype loss = 0;
  switch (method_) {
    case SoftmaxWithDecayLossParameter_Decay_GAUSSIAN:
    forward_cpu_kernel(get_weight_gaussian_cpu, prob_data, label_data,
        weight_data, outer_num_, inner_num_, softmax_dim_, rate_, &loss);
    break;
    case SoftmaxWithDecayLossParameter_Decay_POWER:
    NOT_IMPLEMENTED;
    break;
    default:
    LOG(FATAL) << "Unknown decay method: "
        << method_;
  }
  top[0]->mutable_cpu_data()[0] = loss / (outer_num_ * inner_num_);
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
    const Dtype* weight_data = weights_.cpu_data();
    int count = prob_.count();
    // caffe_mul(count, prob_data, weight_data, bottom_diff);
    caffe_sub(count, prob_data, weight_data, bottom_diff);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / (outer_num_ * inner_num_);
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithDecayLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithDecayLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithDecayLoss);

}  // namespace caffe
