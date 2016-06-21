#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* rand_vec_data = rand_vec_.mutable_gpu_data();
  const int count = bottom[0]->count();
  // generate gaussian noise and add to top, works both for in-place or not

  if (sigma_> 0) {
    caffe_gpu_rng_gaussian(count, Dtype(0), sigma_, rand_vec_data);
    caffe_gpu_add(count, bottom_data, rand_vec_data, top_data);
  } else if (bottom[0]==top[0]) {
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (bottom[0]==top[0]) {
    } else {
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();
      caffe_copy(count, top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);

}  // namespace caffe
