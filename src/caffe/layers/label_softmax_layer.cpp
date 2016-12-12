#include <algorithm>
#include <vector>

#include "caffe/layers/label_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_classes_ = this->layer_param_.label_softmax_param().num_classes();
  label_axis_ = this->layer_param_.label_softmax_param().axis();
}

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> dims = bottom[0]->shape();
  CHECK_EQ(bottom[0]->shape(label_axis_), 1) <<
      "label dimension " << label_axis_ << " should have 1 channel";
  dims[label_axis_] = num_classes_;
  top[0]->Reshape(dims);
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
}

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label = static_cast<int>(bottom_data[i*inner_num_+j]);
      top_data[(i*num_classes_+label)*inner_num_+j] = Dtype(1);
    }
  }
}

template <typename Dtype>
void LabelSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(propagate_down[0], false) << "should not propagate down to label";
}


#ifdef CPU_ONLY
STUB_GPU(LabelSoftmaxLayer);
#endif

INSTANTIATE_CLASS(LabelSoftmaxLayer);
REGISTER_LAYER_CLASS(LabelSoftmax);

}  // namespace caffe
