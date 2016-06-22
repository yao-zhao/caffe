#include <vector>

#include "caffe/layers/ladder_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LadderLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  const Dtype* mean_data = bottom[2]->gpu_data();
  const Dtype* variance_data = mean_data + channels_;
  Dtype* diff_data = diff_.mutable_gpu_data();
  const Dtype* z_hat = bottom[1]->gpu_data();
  const Dtype* z = bottom[0]->gpu_data();
  const Dtype* batch_sum_mul_data = batch_sum_multiplier_.gpu_data();
  const Dtype* spatial_sum_mul_data = spatial_sum_multiplier_.gpu_data();
  Dtype* num_by_chans_data = num_by_chans_.mutable_gpu_data();
  Dtype* tempvar_data = tempvar_.mutable_gpu_data();
  // copy z hat
  caffe_copy(count, z_hat, diff_data);
  // subtract mean from z hat
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_mul_data, mean_data, 0.,
      num_by_chans_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_data,
      spatial_sum_mul_data, 1., diff_data);
  // replicate vairance to input size
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_mul_data, variance_data, 0.,
      num_by_chans_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_data,
      spatial_sum_mul_data, 0., tempvar_data);
  // devide the varaince from z hat
  caffe_gpu_div(count, diff_data, tempvar_data, diff_data);
  // calculate the sub between z and z hat
  caffe_gpu_sub(count, z, diff_data, diff_data);
  // calculate the loss
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void LadderLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype alpha;
    // propagate to z
  if (propagate_down[0]) {
    alpha = top[0]->gpu_diff()[0] / bottom[0]->num();
    caffe_gpu_scale(count, alpha, diff_data, bottom[0]->mutable_gpu_diff());
  }
    // propagate to z hat
  if (propagate_down[1]) {
      // div the diff by variance, modified the diff, cant be used further
    caffe_gpu_div(count, diff_data, tempvar_.gpu_data(), diff_data);
    alpha = - top[0]->gpu_diff()[0] / bottom[0]->num();
    caffe_gpu_scale(count, alpha, diff_data, bottom[1]->mutable_gpu_diff());
  }
    // dont propagate to varibles
}

INSTANTIATE_LAYER_GPU_FUNCS(LadderLossLayer);

}  // namespace caffe
