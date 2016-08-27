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
  CHECK_EQ(inner_num_, 1)
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "H * W must be 1, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(bottom[0]->shape(softmax_axis_), 2)
      << "The number of channels for the softmax has to equal exactly to 2";
  vector<int> shape(4,1);
  shape[0] = outer_num_;
  is_positive_.Reshape(shape);
  is_negative_.Reshape(shape);
  prob_positive_.Reshape(shape);
  prob_negative_.Reshape(shape);
  ones_.Reshape(shape);
  caffe_set(ones_.count(), Dtype(1), ones_.mutable_cpu_data());
  shape[1] = outer_num_;
  diff_.Reshape(shape);
}

template<typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::SetIsPositiveNegative(const int count,
    const Dtype* label) {
  Dtype* is_positive_data = is_positive_.mutable_cpu_data();
  Dtype* is_negative_data = is_negative_.mutable_cpu_data();
  int num_positive = 0;
  int num_negative = 0;
  // set is positive data
  for (int i = 0; i < count; ++i) {
    const int label_value = static_cast<int>(label[i]);
    if (label_value == positive_label_) {
      is_positive_data[i] = 1;
      is_negative_data[i] = 0;
      num_positive++;
    } else if (label_value == negative_label_) {
      is_negative_data[i] = 1;
      is_positive_data[i] = 0;
      num_negative++;
    }
  }
  normalizer_ = num_positive * num_negative;
}

template<typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::ProbToPosNeg() {
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* prob_positive_data = prob_positive_.mutable_cpu_data();
  Dtype* prob_negative_data = prob_negative_.mutable_cpu_data();
  const int count = prob_.count()/2;
  for (int i = 0; i < count; ++i) {
    prob_positive_data[i] = prob_data[2*i+1];
    prob_negative_data[i] = prob_data[2*i];
  }
}

template<typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::PosNegToProb() {
  Dtype* prob_diff = prob_.mutable_cpu_diff();
  const Dtype* prob_positive_diff = prob_positive_.cpu_diff();
  const Dtype* prob_negative_diff = prob_negative_.cpu_diff();
  const int count = prob_.count()/2;
  for (int i = 0; i < count; ++i) {
    prob_diff[2*i+1] = prob_positive_diff[i];
    prob_diff[2*i] = prob_negative_diff[i];
  }
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* ones_data = ones_.cpu_data();
  const int count = bottom[1]->count();
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype* tmp_data = diff_.mutable_cpu_diff();
  // set positive and negative label
  SetIsPositiveNegative(count, label);
  const Dtype* is_positive_data = is_positive_.mutable_cpu_data();
  const Dtype* is_negative_data = is_negative_.mutable_cpu_data();
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  ProbToPosNeg();
  const Dtype* positive_data = prob_positive_.cpu_data();
  const Dtype* negative_data = prob_negative_.cpu_data();
  // calculate diff
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
      Dtype(1), positive_data, ones_data, Dtype(0), diff_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
      Dtype(1), ones_data, negative_data, Dtype(0), tmp_data);
  caffe_sub(count*count, diff_data, tmp_data, diff_data);
  // calculate total AUC
  Dtype auc = 0;
  if (normalizer_ == 0) {
    // if all positive or negative set normalizer to 0, otherwise calculate
    auc = 0.5;
  } else {
    for (int i = 0; i < count; ++i) {
      if (is_positive_data[i] == 0) {
        continue;
      }
      for (int j = 0; j < count; ++j) {
        if (is_negative_data[j] == 0) {
          continue;
        }
        Dtype diff_value = diff_data[i * count + j];
        if (diff_value > eps_) {
          auc += 1;
        } else if (diff_value < -eps_) {
        } else {
          auc += 0.5 + 0.5 * diff_value / eps_;
        }
      }
    }
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
  if (propagate_down[0] && normalizer_ != 0) {
    const int count = bottom[1]->count();
    const Dtype* diff_data = diff_.cpu_data();
    const Dtype* is_positive_data = is_positive_.cpu_data();
    const Dtype* is_negative_data = is_negative_.cpu_data();
    Dtype* positive_diff = prob_positive_.mutable_cpu_diff();
    Dtype* negative_diff = prob_negative_.mutable_cpu_diff();
    Dtype* prob_diff = prob_.mutable_cpu_diff();
    caffe_set(count, Dtype(0), positive_diff);
    caffe_set(count, Dtype(0), negative_diff);
    for (int i = 0; i < count; ++i) {
      if (is_positive_data[i] == 0) {
        continue;
      }
      for (int j = 0; j < count; ++j) {
        if (is_negative_data[j] == 0) {
          continue;
        }
        Dtype diff_value = diff_data[i * count + j];
        if (diff_value < eps_ && diff_value > -eps_) {
          positive_diff[i] -= 1;
          negative_diff[j] += 1;
        }
      }
    }
    PosNegToProb();
    Dtype alpha = top[0]->cpu_diff()[0] * 0.5 / eps_ / normalizer_;
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
