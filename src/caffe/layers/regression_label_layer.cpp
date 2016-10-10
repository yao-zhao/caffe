#include <vector>

#include "caffe/layers/regression_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegressionLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  lb_ = this->layer_param_.regression_label_param().lower_bound();
  ub_ = this->layer_param_.regression_label_param().upper_bound();
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  CHECK_NE(top[1], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void RegressionLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
  top[1]->Reshape(shape);
}

template <typename Dtype>
void RegressionLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data_0 = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top[1]->mutable_cpu_data();

  caffe_set(count, lb_, top_data_0);
  caffe_sub(count, bottom_data, top_data_0, top_data_0);

  caffe_set(count, ub_, top_data_1);
  caffe_sub(count, top_data_1, bottom_data, top_data_1);
}

template <typename Dtype>
void RegressionLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(count, Dtype(0), bottom_diff);
  if (propagate_down[0]) {
    const Dtype* top_diff_0 = top[0]->cpu_diff();
    caffe_add(count, bottom_diff, top_diff_0, bottom_diff);
  }
  if (propagate_down[1]) {
    const Dtype* top_diff_1 = top[1]->cpu_diff();
    caffe_sub(count, bottom_diff, top_diff_1, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(RegressionLabelLayer);
#endif

INSTANTIATE_CLASS(RegressionLabelLayer);
REGISTER_LAYER_CLASS(RegressionLabel);

}  // namespace caffe
