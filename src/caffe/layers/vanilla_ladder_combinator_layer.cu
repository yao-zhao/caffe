#include <cfloat>
#include <vector>

#include "caffe/layers/vanilla_ladder_combinator_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void VanillaLadderCombinatorForward(const int n, 
    const Dtype* bottom_data_z, const Dtype* bottom_data_u,
    const Dtype* weight_b0, const Dtype* weight_w0z, const Dtype* weight_w0u, 
    const Dtype* weight_w0zu, const Dtype* weight_wsigma,
    const Dtype* weight_b1, const Dtype* weight_w1z, const Dtype* weight_w1u,
    const Dtype* weight_w1zu, const int comb_dim, Dtype* top_data, Dtype* tempsig_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    // save the sigmoid calculation to be used later in backprop
    tempsig_data[index] = 1. / (1. + exp( - (weight_b1[comb_index] + 
      weight_w1z[comb_index] * bottom_data_z[index] +
      weight_w1u[comb_index] * bottom_data_u[index] +
      weight_w1zu[comb_index] * bottom_data_z[index] * bottom_data_u[index])));
    top_data[index] = weight_b0[comb_index] + 
      weight_w0z[comb_index] * bottom_data_z[index] +
      weight_w0u[comb_index] * bottom_data_u[index] + 
      weight_w0zu[comb_index] * bottom_data_z[index] * bottom_data_u[index] + 
      weight_wsigma[comb_index] * tempsig_data[index];
  }
}

template <typename Dtype>
__global__ void VanillaLadderCombinatorBackward(const int n, 
    const Dtype* top_diff, Dtype* tempsig_data, 
    const Dtype* bottom_data_z, const Dtype* bottom_data_u,
    const Dtype* weight_w0z, const Dtype* weight_w0u, 
    const Dtype* weight_w0zu, const Dtype* weight_wsigma,
    const Dtype* weight_w1z, const Dtype* weight_w1u,
    const Dtype* weight_w1zu, 
    const int comb_dim, Dtype* bottom_diff_z, Dtype* bottom_diff_u) {
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    // using prestored temsig, assuming that forward is called before backward
    // here tempsig is not modified, inconsistent with cpu version
    bottom_diff_z[index] = top_diff[index] * ( weight_w0z[comb_index] + 
      weight_w0zu[comb_index] * bottom_data_u[index] +
      weight_wsigma[comb_index] * tempsig_data[index] * (1. - tempsig_data[index]) * 
      (weight_w1z[comb_index] + weight_w1zu[comb_index] * bottom_data_u[index]) );
    bottom_diff_u[index] = top_diff[index] * ( weight_w0u[comb_index] + 
      weight_w0zu[comb_index] * bottom_data_z[index] +
      weight_wsigma[comb_index] * tempsig_data[index] * (1. - tempsig_data[index]) * 
      (weight_w1u[comb_index] + weight_w1zu[comb_index] * bottom_data_z[index]) );    
  }
}
// only backpropagate to the second blob, not the first blob
template <typename Dtype>
__global__ void VanillaLadderCombinatorBackward_u(const int n, 
    const Dtype* top_diff, Dtype* tempsig_data, 
    const Dtype* bottom_data_z, 
    const Dtype* weight_w0u, 
    const Dtype* weight_w0zu, const Dtype* weight_wsigma,
    const Dtype* weight_w1u,
    const Dtype* weight_w1zu, 
    const int comb_dim, Dtype* bottom_diff_u) {
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    // using prestored temsig, assuming that forward is called before backward
    // here tempsig is not modified, inconsistent with cpu version
    bottom_diff_u[index] = top_diff[index] * ( weight_w0u[comb_index] + 
      weight_w0zu[comb_index] * bottom_data_z[index] +
      weight_wsigma[comb_index] * tempsig_data[index] * (1. - tempsig_data[index]) * 
      (weight_w1u[comb_index] + weight_w1zu[comb_index] * bottom_data_z[index]) );    
  }
}

// backpropagate to weight blob
template <typename Dtype>
__global__ void VanillaLadderCombinatorBackward_w(const int n,
    const Dtype* top_diff, const Dtype* tempmul_data, Dtype* tempsig_data, 
    const Dtype* bottom_data_z, const Dtype* bottom_data_u, 
    const Dtype* weight_wsigma, const int comb_dim, 
    Dtype* weight_diff_b0, Dtype* weight_diff_w0z, 
    Dtype* weight_diff_w0u, Dtype* weight_diff_w0zu,
    Dtype* weight_diff_wsigma, 
    Dtype* weight_diff_b1, Dtype* weight_diff_w1z,
    Dtype* weight_diff_w1u, Dtype* weight_diff_w1zu){
  CUDA_KERNEL_LOOP(index, n) {
    const int comb_index = index % comb_dim;
    weight_diff_b0[comb_index] += top_diff[index];
    weight_diff_w0z[comb_index] += top_diff[index] * bottom_data_z[index];
    weight_diff_w0u[comb_index] += top_diff[index] * bottom_data_u[index];
    weight_diff_w0zu[comb_index] += top_diff[index] * bottom_data_z[index] * bottom_data_u[index];
    weight_diff_wsigma[comb_index] += tempsig_data[index];
    tempsig_data[index] = tempsig_data[index] * ( 1 - tempsig_data[index] ) 
    * weight_wsigma[comb_index] * top_diff[index]; 
    weight_diff_b1[comb_index] += tempsig_data[index];
    weight_diff_w1z[comb_index] += tempsig_data[index] * bottom_data_z[index];
    weight_diff_w1u[comb_index] += tempsig_data[index] * bottom_data_u[index];
    weight_diff_w1zu[comb_index] += tempsig_data[index] * bottom_data_z[index] * bottom_data_u[index]; 
  }
}

// calculate the differnce of sigmored 
template <typename Dtype>
__global__ void diff_sigmoid(const int n, Dtype* sig) {
  CUDA_KERNEL_LOOP(index, n) {
    sig[index] = sig[index] * (1-sig[index]);
  }
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // init
  const Dtype* bottom_data_z = bottom[0]->gpu_data();
  const Dtype* bottom_data_u = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* tempsig_data = tempsig_.mutable_gpu_data();
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

  VanillaLadderCombinatorForward<Dtype>  
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data_z, bottom_data_u, 
    weight_b0, weight_w0z, weight_w0u, weight_w0zu, weight_wsigma,
    weight_b1, weight_w1z, weight_w1u, weight_w1zu,
    comb_dim_, top_data, tempsig_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data_z = bottom[0]->gpu_data();
  const Dtype* bottom_data_u = bottom[1]->gpu_data();
  Dtype* tempsig_data = tempsig_.mutable_gpu_data();
  Dtype* tempmul_data = tempmul_.mutable_gpu_data();
  const Dtype* sum_mult = sum_multiplier_.gpu_data();
  const Dtype* weight_w0z = this->blobs_[1]->gpu_data();
  const Dtype* weight_w0u = this->blobs_[2]->gpu_data();
  const Dtype* weight_w0zu = this->blobs_[3]->gpu_data();
  const Dtype* weight_wsigma = this->blobs_[4]->gpu_data();
  const Dtype* weight_w1z = this->blobs_[6]->gpu_data();
  const Dtype* weight_w1u = this->blobs_[7]->gpu_data();
  const Dtype* weight_w1zu = this->blobs_[8]->gpu_data();
  Dtype* bottom_diff_u = bottom[1]->mutable_gpu_diff();
  Dtype* weight_diff_b0 = this->blobs_[0]->mutable_gpu_diff();    
  Dtype* weight_diff_w0z = this->blobs_[1]->mutable_gpu_diff();
  Dtype* weight_diff_w0u = this->blobs_[2]->mutable_gpu_diff();
  Dtype* weight_diff_w0zu = this->blobs_[3]->mutable_gpu_diff();
  Dtype* weight_diff_wsigma = this->blobs_[4]->mutable_gpu_diff();
  Dtype* weight_diff_b1 = this->blobs_[5]->mutable_gpu_diff();
  Dtype* weight_diff_w1z = this->blobs_[6]->mutable_gpu_diff();
  Dtype* weight_diff_w1u = this->blobs_[7]->mutable_gpu_diff();
  Dtype* weight_diff_w1zu = this->blobs_[8]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  // make sure calculate this first before param diff and make sure that tempsig_ is not modified during calculation
  CHECK(!(propagate_down[0]==1 && propagate_down[1]==0)) <<
   "currently only support propagate down to at least the second bottoms"
   << " propagate_down[0]="<<propagate_down[0]
   << " propagate_down[1]="<<propagate_down[1]
   << " current layer name: "<<this->layer_param_.name();
  if (propagate_down[0] && propagate_down[1]) {
    Dtype* bottom_diff_z = bottom[0]->mutable_gpu_diff();
    VanillaLadderCombinatorBackward<Dtype>
    <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, tempsig_data, 
      bottom_data_z, bottom_data_u, 
      weight_w0z, weight_w0u, weight_w0zu, weight_wsigma,
      weight_w1z, weight_w1u, weight_w1zu,
      comb_dim_, bottom_diff_z, bottom_diff_u);
    CUDA_POST_KERNEL_CHECK;
  } else if (propagate_down[1]) {
    VanillaLadderCombinatorBackward_u<Dtype>
    <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, tempsig_data, 
      bottom_data_z, 
      weight_w0u, weight_w0zu, weight_wsigma,
      weight_w1u, weight_w1zu,
      comb_dim_, bottom_diff_u);
    CUDA_POST_KERNEL_CHECK;
  }

  // this wont work because multiple cores are accessing same location causing unsynchronized read and write
  // VanillaLadderCombinatorBackward_w<Dtype>
  // <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //   count, top_diff, 
  //   tempmul_data, tempsig_data, 
  //   bottom_data_z, bottom_data_u, 
  //   weight_wsigma, comb_dim_, 
  //   weight_diff_b0, weight_diff_w0z, 
  //   weight_diff_w0u, weight_diff_w0zu,
  //   weight_diff_wsigma, 
  //   weight_diff_b1, weight_diff_w1z,
  //   weight_diff_w1u, weight_diff_w1zu );
  // CUDA_POST_KERNEL_CHECK;

  // update param for backprop
  // hack: use tempmul_ to hold temporary result
  // hack: use tempsig_ to hold temporary sig result, tempsig = top_diff* wsig * Sig
  // weights should be accumulated, they will be cleared every step by solver

  //b0
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), top_diff, sum_mult, Dtype(1), weight_diff_b0);
  // w0z 
  caffe_gpu_mul<Dtype>(count, top_diff, bottom_data_z, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w0z);
  // w0u 
  caffe_gpu_mul<Dtype>(count, top_diff, bottom_data_u, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w0u);
    // w0zu 
  caffe_gpu_mul<Dtype>(count, top_diff, bottom_data_z, tempmul_data);
  caffe_gpu_mul<Dtype>(count, bottom_data_u, tempmul_data, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w0zu);
  // wsigma
  caffe_gpu_mul<Dtype>(count, top_diff, tempsig_data, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_wsigma);    
  // modify tempsig
  diff_sigmoid<Dtype>
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, tempsig_data);
  caffe_gpu_mul<Dtype>(count, top_diff, tempsig_data, tempsig_data);
  // b1
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempsig_data, sum_mult, Dtype(1), weight_diff_b1);
  caffe_gpu_mul<Dtype>(comb_dim_, weight_wsigma, weight_diff_b1, weight_diff_b1);
    // w0z 
  caffe_gpu_mul<Dtype>(count, tempsig_data, bottom_data_z, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w1z);
  caffe_gpu_mul<Dtype>(comb_dim_, weight_wsigma, weight_diff_w1z, weight_diff_w1z);
    // w0u 
  caffe_gpu_mul<Dtype>(count, tempsig_data, bottom_data_u, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w1u);
  caffe_gpu_mul<Dtype>(comb_dim_, weight_wsigma, weight_diff_w1u, weight_diff_w1u);
    // w0zu 
  caffe_gpu_mul<Dtype>(count, tempsig_data, bottom_data_z, tempmul_data);
  caffe_gpu_mul<Dtype>(count, bottom_data_u, tempmul_data, tempmul_data);
  caffe_gpu_gemv<Dtype>(CblasTrans, outer_dim_, comb_dim_,
    Dtype(1), tempmul_data, sum_mult, Dtype(1), weight_diff_w1zu);
  caffe_gpu_mul<Dtype>(comb_dim_, weight_wsigma, weight_diff_w1zu, weight_diff_w1zu);

}

INSTANTIATE_LAYER_GPU_FUNCS(VanillaLadderCombinatorLayer);

}  // namespace caffe
