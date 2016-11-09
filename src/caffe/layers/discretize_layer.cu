#include <vector>

#include "caffe/layers/discretize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DiscretizeForward(const int n, const int num_separators,
    const Dtype* separators_data, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(i, n) {
    if (bottom_data[i] > separators_data[num_separators-1]) {
      top_data[i] = Dtype(num_separators);
    } else {
      for (int j = 0; j < num_separators; ++j) {
        if (bottom_data[i] <= separators_data[j]) {
          top_data[i] = Dtype(j);
          break;
        }
      }
    }
  }
}

template <typename Dtype>
void DiscretizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* separators_data = separators_.gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DiscretizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, num_separators_, separators_data, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  if (top.size() >= 2) {
    Dtype* centers_data = top[1]->mutable_cpu_data();
    centers_data[0] = separators_data[0] * Dtype(1.5) -
        separators_data[1] * Dtype(0.5);
    centers_data[num_separators_] =
        separators_data[num_separators_-1] * Dtype(1.5) -
        separators_data[num_separators_-2] * Dtype(0.5);
    for (int i = 0; i < num_separators_-1; ++i) {
      centers_data[i+1] =
          (separators_data[i] + separators_data[i+1]) / Dtype(2);
    }
  }
}


INSTANTIATE_LAYER_GPU_FORWARD(DiscretizeLayer);


}  // namespace caffe
