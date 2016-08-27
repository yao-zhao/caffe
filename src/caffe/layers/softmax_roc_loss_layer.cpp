#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_roc_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  positive_label_ =
      this->layer_param_.softmax_roc_loss_param().positive_label();
  negative_label_ =
      this->layer_param_.softmax_roc_loss_param().negative_label();
  eps_ = this->layer_param_.softmax_roc_loss_param().eps();
  CHECK_GT(eps_, 0) << "linear region must be larger than zero";
  CHECK_LT(eps_, 0.5) << "linear region must be lesser than 0.5";
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(inner_num_,1)
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "H * W must be 1, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(bottom[0]->shape(softmax_axis_), 2)
      << "The number of channels for the softmax has to equal exactly to 2";
  is_positive_.Reshape(outer_num_, 1, 1, 1);
  is_negative_.Reshape(outer_num_, 1, 1, 1);
  ones_.ReshapeLike(*bottom[1]);
  caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
  diff_.Reshape(outer_num_, outer_num_, 1, 1);
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* ones_data = ones_.cpu_data();
  const int count = bottom[1]->count();
  const Dtype* positive_data = prob_.cpu_data()+count;
  const Dtype* negative_data = prob_.cpu_data();
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype* tmp_data = diff_.mutable_cpu_diff();
  bool* is_positive_data = is_positive_.mutable_cpu_data();
  bool* is_negative_data = is_negative_.mutable_cpu_data();
  // set is positive data
  for (int i = 0; i < count; ++i) {
    const int label_value = static_cast<int>(label[i]);
    is_positive_data[i] = label_value == positive_label_;
    is_negative_data[i] = label_value == negative_label_;
  }
  // calculate diff
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, count, 1, count,
      Dtype(1), positive_data, ones_data, Dtype(1), diff_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, count, 1, count,
      Dtype(1), ones_data, negative_data, Dtype(1), tmp_data);
  caffe_sub(count*count, diff_data, tmp_data, diff_data);
  // calculate total AUC
  Dtype auc = 0;
  normalizer_ = 0;
  for (int i = 0; i < count; ++i) {
    if (!is_positive_data[i]) {
      continue;
    }
    for (int j = 0; j < count; ++j) {
      if (!is_negative_data[i]) {
        continue;
      }
      normalizer_++;
      Dtype diff_value = diff_data[i * count + j];
      if (diff_value > eps_) {
        auc += 1;
      } else if (diff_value < -eps_) {
      } else {
        auc += 0.5 + 0.5 * diff_value / eps_;
      }
    }
  }
  // if all positive or negative set normalizer to 0, otherwise calculate
  if (normalizer_ == 0) {
    auc = 0.5;
  } else {
    auc /= normalizer_;
  }
  // optimize 1-auc
  top[0]->mutable_cpu_data()[0] = 1-auc;
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[1]->count();
    const Dtype* diff_data = diff_.cpu_data();
    const bool* is_positive_data = is_positive_.cpu_data();
    const bool* is_negative_data = is_negative_.cpu_data();
    Dtype* positive_diff = prob_.mutable_cpu_diff()+count;
    Dtype* negative_diff = prob_.mutable_cpu_diff();
    Dtype* prob_diff = prob_.mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      if (!is_positive_data[i]) {
        continue;
      }
      for (int j = 0; j < count; ++j) {
        if (!is_negative_data[i]) {
          continue;
        }
        Dtype diff_value = diff_data[i * count + j];
        if (diff_value < eps_ && diff_value > -eps_) {
          positive_diff[i] -= 1;
          negative_diff[i] += 1;
        }
      }
    }
    Dtype alpha = top[0]->cpu_diff()[0] * Dtype(0.5) / eps_ / normalizer_;
    caffe_scal(2*count, alpha, prob_diff);
    softmax_layer_->Backward(softmax_top_vec_, propagate_down,
        softmax_bottom_vec_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithROCLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithROCLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithROCLoss);

}  // namespace caffe
