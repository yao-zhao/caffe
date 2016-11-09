#include <vector>

#include "caffe/layers/discretize_layer.hpp"

namespace caffe {

template <typename Dtype>
void DiscretizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const DiscretizeParameter& param = this->layer_param_.discretize_param();
  num_separators_ = param.separator_size();
  CHECK_GT(num_separators_, 0) << "number of separators must be greater than 0";
  separators_.Reshape(*(new vector<int>(1, num_separators_)));
  Dtype* separators_data = separators_.mutable_cpu_data();
  for (int i = 0; i < num_separators_; ++i) {
    separators_data[i] = param.separator(i);
  }
}

template <typename Dtype>
void  DiscretizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    top[1]->Reshape(*(new vector<int>(1, num_separators_ + 1)));
  }
}

template <typename Dtype>
void DiscretizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* separators_data = separators_.cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] > separators_data[num_separators_-1]) {
      top_data[i] = Dtype(num_separators_);
    } else {
      for (int j = 0; j < num_separators_; ++j) {
        if (bottom_data[i] <= separators_data[j]) {
          top_data[i] = Dtype(j);
          break;
        }
      }
    }
  }
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

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DiscretizeLayer, Forward);
#endif

INSTANTIATE_CLASS(DiscretizeLayer);
REGISTER_LAYER_CLASS(Discretize);

}  // namespace caffe
