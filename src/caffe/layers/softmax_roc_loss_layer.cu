#include <glog/logging.h>

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_roc_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxROCLossForwardGPU(const int nthreads,
          const Dtype* diff_data,
          const Dtype eps, Dtype* is_positive_negative_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (is_positive_negative_data[index] != 0) {
      if (diff_data[index] > eps) {
        is_positive_negative_data[index] = 1;
      } else if (diff_data[index] > -eps) {
        is_positive_negative_data[index] = 0.5 + 0.5 * diff_data[index] / eps;
      } else {
        is_positive_negative_data[index] = 0;
      }
    }
  }
}

template <typename Dtype>
__global__ void ProbToPosNegGPU(const int nthreads,
          const Dtype* prob_data,
          Dtype* pos_data, Dtype* neg_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    pos_data[index] = prob_data[2*index+1];
    neg_data[index] = prob_data[2*index];
  }
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* ones_data = ones_.gpu_data();
  const int count = bottom[1]->count();
  const int count2 = count * count;
  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype* tmp_data = diff_.mutable_gpu_diff();
  // set is positive data
  SetIsPositiveNegative(count, bottom[1]->cpu_data());
  const Dtype* is_positive_data = is_positive_.mutable_gpu_data();
  const Dtype* is_negative_data = is_negative_.mutable_gpu_data();
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ProbToPosNegGPU<Dtype> <<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, prob_.gpu_data(),
      prob_positive_.mutable_gpu_data(), prob_negative_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  const Dtype* positive_data = prob_positive_.gpu_data();
  const Dtype* negative_data = prob_negative_.gpu_data();
  // calculate diff
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
      Dtype(1), positive_data, ones_data, Dtype(0), diff_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
      Dtype(1), ones_data, negative_data, Dtype(0), tmp_data);
  caffe_gpu_sub(count*count, diff_data, tmp_data, diff_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
      Dtype(1), is_positive_data, is_negative_data, Dtype(0), tmp_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxROCLossForwardGPU<Dtype> <<<CAFFE_GET_BLOCKS(count2),
      CAFFE_CUDA_NUM_THREADS>>>(count2, diff_data, eps_, tmp_data);
  CUDA_POST_KERNEL_CHECK;
  // calculate auc
  Dtype auc = 0;
  if (normalizer_ == 0) {
    // if all positive or negative set normalizer to 0, otherwise calculate
    auc = 0.5;
  } else {
    caffe_gpu_asum(count2, tmp_data, &auc);
    auc /= normalizer_;
  }
  // optimize 1-auc
  top[0]->mutable_cpu_data()[0] = 1-auc;
}

template <typename Dtype>
__global__ void SoftmaxROCLossBackwardGPU(const int nthreads,
          const Dtype* diff_data, const Dtype eps,
          Dtype* is_positive_negative_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (is_positive_negative_data[index] != 0 &&
        (diff_data[index] > eps || diff_data[index] < -eps)) {
      is_positive_negative_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void PosNegToProbGPU(const int nthreads,
          const Dtype* pos_data, const Dtype* neg_data,
          Dtype* prob_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    prob_data[2*index+1] = pos_data[index];
    prob_data[2*index] = neg_data[index];
  }
}

template <typename Dtype>
void SoftmaxWithROCLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0] && normalizer_ != 0) {
    const int count = bottom[1]->count();
    const int count2 = count * count;
    const Dtype* is_positive_data = is_positive_.gpu_data();
    const Dtype* is_negative_data = is_negative_.gpu_data();
    const Dtype* diff_data = diff_.gpu_data();
    const Dtype alpha = top[0]->cpu_diff()[0] * 0.5 / eps_ / normalizer_;
    Dtype* tmp_data = diff_.mutable_gpu_diff();
    Dtype* positive_diff = prob_positive_.mutable_gpu_diff();
    Dtype* negative_diff = prob_negative_.mutable_gpu_diff();
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, count, count, 1,
        Dtype(1), is_positive_data, is_negative_data, Dtype(0), tmp_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxROCLossBackwardGPU<Dtype> <<<CAFFE_GET_BLOCKS(count2),
        CAFFE_CUDA_NUM_THREADS>>>(count2, diff_data, eps_, tmp_data);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, count, 1, count,
        -alpha, tmp_data, is_negative_data, Dtype(0), positive_diff);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, count, count,
        alpha, is_positive_data, tmp_data, Dtype(0), negative_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PosNegToProbGPU<Dtype> <<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, positive_diff, negative_diff,
        prob_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    softmax_layer_->Backward(softmax_top_vec_, propagate_down,
        softmax_bottom_vec_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithROCLossLayer);

}  // namespace caffe
