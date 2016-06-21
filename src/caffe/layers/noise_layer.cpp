// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  sigma_ = this->layer_param_.noise_param().sigma();
  CHECK(sigma_ > 0.) << "noise level has to be greater than zero";
//  CHECK(sigma_ < 3.) << "noise level has to be lesser than 3";
}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]==top[0]) {
  } else {
    top[0]->ReshapeLike(*bottom[0]);
  }
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* rand_vec_data = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  // create gaussian noise and add to top, in-place/ or not the same
  if (sigma_> 0) {
    caffe_rng_gaussian(count, Dtype(0), sigma_, rand_vec_data);
    caffe_add(count, rand_vec_data, bottom_data, top_data);
  } else if (bottom[0]==top[0]) {
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (bottom[0]==top[0]) {      
      // in-place doing nothing
    } else {
      // copy top diff to bot diff
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const int count = bottom[0]->count();
      caffe_copy(count, top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);

}  // namespace caffe
