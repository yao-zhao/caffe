#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/label_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void label_softmax_forward_kernel(const int count,
    const int channels, const int inner_dim, const Dtype* label, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / inner_dim;
    int s = index % inner_dim;
    out[(n*channels+static_cast<int>(label[index]))+s] = 1;
  }
}

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = bottom[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  label_softmax_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, num_classes_, inner_num_,
      bottom_data, top_data);
}

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(propagate_down[0], false) << "should not propagate down to label";
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelSoftmaxLayer);


}  // namespace caffe
