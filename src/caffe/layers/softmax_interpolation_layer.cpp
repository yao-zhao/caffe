#include <vector>

#include "caffe/layers/softmax_interpolation_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxInterpolationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_interpolation_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
}

template <typename Dtype>
void  SoftmaxInterpolationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(softmax_axis_), bottom[1]->count()) <<
      "the dimension of the scond bottom and" <<
      "the dimension of the softmax axis must be equal";
  vector<int> top_dims = bottom[0]->shape();
  top_dims[softmax_axis_] = 1;
  top[0]->Reshape(top_dims);
}

template <typename Dtype>
void SoftmaxInterpolationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* interpolation_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // use top_diff to hold temporary data
  Dtype* tmp_data = top[0]->mutable_cpu_diff();
  const int softmax_dim = bottom[1]->count();
  for (int i = 0; i < outer_num_; ++i) {
    caffe_set(inner_num_, Dtype(0), top_data);
    for (int j = 0; j < softmax_dim; ++j) {
      caffe_cpu_scale(inner_num_, interpolation_data[j], bottom_data, tmp_data);
      caffe_add(inner_num_, top_data, tmp_data, top_data);
      bottom_data += inner_num_;
    }
    top_data += inner_num_;
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(SoftmaxInterpolationLayer, Forward);
#endif

INSTANTIATE_CLASS(SoftmaxInterpolationLayer);
REGISTER_LAYER_CLASS(SoftmaxInterpolation);

}  // namespace caffe
