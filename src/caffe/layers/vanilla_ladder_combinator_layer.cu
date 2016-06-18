#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void VanillaLadderCombinatorForward(const int n, 
    const Dtype* bottom_data_z, const Dtype* bottom_data_u,
    const Dtype* weight_b0, const Dtype* weight_w0z, const Dtype* weight_w0u, 
    const Dtype* weight_w0zu, const Dtype* weight_wsigma,
    const Dtype* weight_b1, const Dtype* weight_w1z, const Dtype* weight_w1u,
    const Dtype* weight_w1zu, const int comb_dim, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    top_data[index] = weight_b0[comb_index] + 
      weight_w0z[comb_index] * bottom_data_z[index] +
      weight_w0u[comb_index] * bottom_data_u[index] + 
      weight_w0zu[comb_index] * bottom_data_z[index] * bottom_data_u[index] + 
      weight_wsigma[comb_index] * 1. / (1. + exp( - (weight_b1[comb_index] + 
      weight_w1z[comb_index] * bottom_data_z[index] +
      weight_w1u[comb_index] * bottom_data_u[index] +
      weight_w1zu[comb_index] * bottom_data_z[index] * bottom_data_u[index])));
  }
}

template <typename Dtype>
__global__ void VanillaLadderCombinatorBackward(const int n, 
    const Dtype* bottom_data_z, const Dtype* bottom_data_u,
    const Dtype* weight_w0z, const Dtype* weight_w0u, 
    const Dtype* weight_w0zu, const Dtype* weight_wsigma,
    const Dtype* weight_w1z, const Dtype* weight_w1u,
    const Dtype* weight_w1zu, 
    Dtype* bottom_diff_z, Dtype* bottom_diff_u, 
    Dtype* weight_diff_b0, Dtype* weight_diff_w0z, Dtype* weight_diff_w0u, 
    Dtype* weight_diff_w0zu, Dtype* weight_diff_wsigma,
    Dtype* weight_diff_b1, Dtype* weight_diff_w1z, Dtype* weight_diff_w1u,
    Dtype* weight_diff_w1zu, 
    const int comb_dim, Dtype* top_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    top_data[index] = weight_b0[comb_index] + 
      weight_w0z[comb_index] * bottom_data_z[index] +
      weight_w0u[comb_index] * bottom_data_u[index] + 
      weight_w0zu[comb_index] * bottom_data_z[index] * bottom_data_u[index] + 
      weight_wsigma[comb_index] * 1. / (1. + exp( - (weight_b1[comb_index] + 
      weight_w1z[comb_index] * bottom_data_z[index] +
      weight_w1u[comb_index] * bottom_data_u[index] +
      weight_w1zu[comb_index] * bottom_data_z[index] * bottom_data_u[index])));
  }
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // init
  const Dtype* bottom_data_z = bottom[0]->gpu_data();
  const Dtype* bottom_data_u = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_b0 = this->blobs_[0]->gpu_data();
  const Dtype* weight_w0z = this->blobs_[1]->gpu_data();
  const Dtype* weight_w0u = this->blobs_[2]->gpu_data();
  const Dtype* weight_w0zu = this->blobs_[3]->gpu_data();
  const Dtype* weight_wsigma = this->blobs_[4]->gpu_data();
  const Dtype* weight_b1 = this->blobs_[5]->gpu_data();
  const Dtype* weight_w1z = this->blobs_[6]->gpu_data();
  const Dtype* weight_w1u = this->blobs_[7]->gpu_data();
  const Dtype* weight_w1zu = this->blobs_[8]->gpu_data();
  const int count = top[0]->count();
  // Dtype* temp_data = temp_.mutable_gpu_data();
  // Dtype* tempsig_data = tempsig_.mutable_gpu_data();
  // Dtype* tempmul_data = tempmul_.mutable_gpu_data();

  VanillaLadderCombinatorForward<Dtype>  
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data_z, bottom_data_u, 
    weight_b0, weight_w0z, weight_w0u, weight_w0zu, weight_wsigma,
    weight_b1, weight_w1z, weight_w1u, weight_w1zu,
    comb_dim_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data_z = bottom[0]->gpu_data();
    const Dtype* bottom_data_u = bottom[1]->gpu_data();
    const Dtype* weight_w0z = this->blobs_[1]->gpu_data();
    const Dtype* weight_w0u = this->blobs_[2]->gpu_data();
    const Dtype* weight_w0zu = this->blobs_[3]->gpu_data();
    const Dtype* weight_wsigma = this->blobs_[4]->gpu_data();
    const Dtype* weight_w1z = this->blobs_[6]->gpu_data();
    const Dtype* weight_w1u = this->blobs_[7]->gpu_data();
    const Dtype* weight_w1zu = this->blobs_[8]->gpu_data();
    Dtype* bottom_diff_z = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_u = bottom[1]->mutable_gpu_diff();
    Dtype* weight_diff_b0 = this->blobs_[0]->mutable_gpu_data();    
    Dtype* weight_diff_w0z = this->blobs_[1]->mutable_gpu_data();
    Dtype* weight_diff_w0u = this->blobs_[2]->mutable_gpu_data();
    Dtype* weight_diff_w0zu = this->blobs_[3]->mutable_gpu_data();
    Dtype* weight_diff_wsigma = this->blobs_[4]->mutable_gpu_data();
    Dtype* weight_diff_b1 = this->blobs_[5]->mutable_gpu_data();
    Dtype* weight_diff_w1z = this->blobs_[6]->mutable_gpu_data();
    Dtype* weight_diff_w1u = this->blobs_[7]->mutable_gpu_data();
    Dtype* weight_diff_w1zu = this->blobs_[8]->mutable_gpu_data();
    const int count = bottom[0]->count();

    VanillaLadderCombinatorBackward<Dtype>
    <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_z, bottom_data_u, 
      weight_w0z, weight_w0u, weight_w0zu, weight_wsigma,
      weight_w1z, weight_w1u, weight_w1zu,
      bottom_diff_z, bottom_diff_u, 
      weight_diff_b0, weight_diff_w0z, weight_diff_w0u, weight_diff_w0zu,
      weight_diff_wsigma,
      weight_diff_b1, weight_diff_w1z, weight_diff_w1u, weight_diff_w1zu,
      comb_dim_, top_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(VanillaLadderCombinatorLayer);

}  // namespace caffe
