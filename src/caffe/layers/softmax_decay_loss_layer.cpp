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
  num_classes_ = bottom[0]->shape(softmax_axis_);
  CHECK_NE(has_ignore_label_, true) << "doesn't alow ignore label";
  CHECK_NE(weight_by_label_freqs_, true) << "doesn't alow label frequency"
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::Reshape(bottom, top);
  weights_.ReshapeLike(*bottom[0]);
}

// template <typename Dtype>
// inline Dtype get_weight_gaussian_cpu(int diff_label) {
//   return exp(-(diff_label*diff_label)/2);
// }

// template <typename Dtype>
// inline void forward_cpu_kernel(Dtype(*weight_func)(int),
//     const Dtype* prob_data, const Dtype* label_data, Dtype* weight_data,
//     const int outer_num, const int inner_num, const int dim,
//     Dtype* loss, Dtype* count) {
//   for (int i = 0; i < outer_num; ++i) {
//     for (int j = 0; j < inner_num; j++) {
//       const int label_value = static_cast<int>(label_data[i * inner_num + j]);
//       Dtype sum_weights = 0;
//       Dtype sum_loss = 0;
//       for (int l = 0; l < dim; ++l) {
//         const int idx = i * dim + label_value * inner_num + j;
//         const Dtype weight = weight_func(l - label_value);
//         sum_weights += weight;
//         sum_loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN))) * weight;
//         weight_data[idx] = weight;
//       }
//       if (sum_weights > 0) {
//         loss += sum_loss/sum_weights;
//         for (int l = 0; l < dim; ++l) {
//           const int idx = i * dim + label_value * inner_num + j;
//           weight_data[idx] /= sum_weights;
//         }
//         ++count;
//       }
//     }
//   }
// }

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight_data = weights_.cpu_data();
  int dim = prob_.count() / outer_num_;
  const int count = 0;

  switch (method_) {
    case SoftmaxWithDecayLossParameter_Decay_GAUSSIAN:
    weight_data = caffe_sub
    // forward_cpu_kernel(&get_weight_gaussian_cpu, prob_data, label_data,
    //     weight_data, outer_num_, inner_num_, dim, &loss, &count) 
    break;
    case SoftmaxWithDecayLossParameter_Decay_POWER:
    NOT_IMPLEMENTED;
    break;
    default:
    LOG(FATAL) << "Unknown decay method: "
        << SoftmaxWithDecayLossParameter_Decay_Name(method_);
  }
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
