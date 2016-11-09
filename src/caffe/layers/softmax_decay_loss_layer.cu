#include <glog/logging.h>

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_decay_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SoftmaxGaussianWeightForwardGPU(const int nthreads,
    const Dtype* label_data, const Dtype rate,
    const int outer_num, const int inner_num, const int dim,
    Dtype* weight_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (inner_num * dim);
    const int j = index % inner_num;
    const int l = (index / inner_num) % dim;
    Dtype label_diff = l - label_data[j + inner_num * i];
    weight_data[index] = exp(-(label_diff * label_diff) / (rate * rate));
  }
}


template <typename Dtype>
__global__ void SoftmaxDecayLossForwardGPU(const int nthreads,
    const Dtype* prob_data, const Dtype* weight_data,
    Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    loss_data[index] = -log(max(prob_data[index], Dtype(FLT_MIN)))
        * weight_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* weight_data = weights_.mutable_gpu_data();
  const int count = prob_.count();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // use label diff to save sum of weights
  Dtype* weight_sum_data = bottom[1]->mutable_gpu_diff();
  // loss
  Dtype loss;
  // calculate weight using different method
  switch (method_) {
    case SoftmaxWithDecayLossParameter_Decay_GAUSSIAN:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxGaussianWeightForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, label_data, rate_,
        outer_num_, inner_num_, softmax_dim_, weight_data);
    break;
    case SoftmaxWithDecayLossParameter_Decay_POWER:
    NOT_IMPLEMENTED;
    break;
    default:
    LOG(FATAL) << "Unknown decay method: "
        << method_;
  }
  CUDA_POST_KERNEL_CHECK;
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, softmax_dim_, inner_num_,
      weight_data, weight_sum_data);
  CUDA_POST_KERNEL_CHECK;
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, softmax_dim_, inner_num_,
      weight_sum_data, weight_data);
  CUDA_POST_KERNEL_CHECK;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxDecayLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, prob_data, weight_data, loss_data);
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_asum(count, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / (outer_num_ * inner_num_);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithDecayLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* weight_data = weights_.gpu_data();
    int count = prob_.count();
    // caffe_gpu_mul(count, prob_data, weight_data, bottom_diff);
    caffe_gpu_sub(count, prob_data, weight_data, bottom_diff);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / (outer_num_ * inner_num_);
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithDecayLossLayer);

}  // namespace caffe
