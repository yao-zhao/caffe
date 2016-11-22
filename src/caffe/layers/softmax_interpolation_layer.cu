#include <vector>

#include "caffe/layers/softmax_interpolation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxInterpolationGPUForward(const int n,
    const Dtype* bottom_data, const Dtype* interpolation_data,
    const int softmax_dim, const int inner_dim,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int i = index / inner_dim;
    const int j = index % inner_dim;
    Dtype sum = 0;
    for (int c = 0; c < softmax_dim; c++) {
      sum += bottom_data[(i * softmax_dim + c) * inner_dim + j] *
          interpolation_data[c];
    }
    top_data[index] = sum;
  }
}

template <typename Dtype>
void SoftmaxInterpolationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* interpolation_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int softmax_dim = bottom[1]->count();
  const int top_count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxInterpolationGPUForward<Dtype><<<CAFFE_GET_BLOCKS(top_count),
      CAFFE_CUDA_NUM_THREADS>>>(top_count, bottom_data, interpolation_data,
      softmax_dim, inner_num_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(SoftmaxInterpolationLayer);

}  // namespace caffe
