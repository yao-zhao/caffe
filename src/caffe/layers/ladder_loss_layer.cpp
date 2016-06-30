#include <vector>

#include "caffe/layers/ladder_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LadderLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // add weight to 1 if no weight is specified
  LossLayer<Dtype>::Reshape(bottom, top);
  // check dimension
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  << "Inputs must have the same dimension.";

// initialize diff
  diff_.ReshapeLike(*bottom[0]);
  tempvar_.ReshapeLike(*bottom[0]);

// intialize batchnorm param
  if (bottom.size()>2) {
    CHECK_EQ(bottom[2]->shape(0), 2) <<"first dimension of batch norm param has to be 2";
    CHECK_EQ(bottom[2]->shape(1), bottom[0]->shape(1)) <<"channel dimension has to agree";

    vector<int> sz;
    channels_ = bottom[0]->shape(1);
    sz.push_back(bottom[0]->shape(0));
    batch_sum_multiplier_.Reshape(sz);

  // allocate space to store vector of 1s that is spatialdim \times 1
    int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
    if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
      sz[0] = spatial_dim;
      spatial_sum_multiplier_.Reshape(sz);
      Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
      caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
    }

    // allocate a file to store by channel result, allocate for batch sum
    int numbychans = channels_*bottom[0]->shape(0);
    if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
      sz[0] = numbychans;
      num_by_chans_.Reshape(sz);
      caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
    }  
  }
}

template <typename Dtype>
void LadderLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  Dtype* diff_data = diff_.mutable_cpu_data();
  const Dtype* z_hat = bottom[1]->cpu_data();
  const Dtype* z = bottom[0]->cpu_data();
  Dtype* tempvar_data = tempvar_.mutable_cpu_data();

  if (bottom.size()>2) {
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
    const Dtype* mean_data = bottom[2]->cpu_data();
    const Dtype* variance_data = mean_data + channels_;
    const Dtype* batch_sum_mul_data = batch_sum_multiplier_.cpu_data();
    const Dtype* spatial_sum_mul_data = spatial_sum_multiplier_.cpu_data();
    Dtype* num_by_chans_data = num_by_chans_.mutable_cpu_data();
    // copy z hat
    caffe_copy(count, z_hat, diff_data);
    // subtract mean from z hat
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_mul_data, mean_data, 0.,
      num_by_chans_data);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, -1, num_by_chans_data,
      spatial_sum_mul_data, 1., diff_data);
    // replicate vairance to input size
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_mul_data, variance_data, 0.,
      num_by_chans_data);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_data,
      spatial_sum_mul_data, 0., tempvar_data);
    // devide the varaince from z hat
    caffe_div(count, diff_data, tempvar_data, diff_data);
    // calculate the sub between z and z hat
    caffe_sub(count, z, diff_data, diff_data);
  } else {
    // calculate without normalization
    caffe_sub(count, z, z_hat, diff_data);
  }
  // calculate the loss
  Dtype dot = caffe_cpu_dot(count, diff_data, diff_data);
  Dtype loss = dot / bottom[0]->count() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LadderLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype alpha;
    // propagate to z
    if (propagate_down[0]) {
      alpha = top[0]->cpu_diff()[0] / bottom[0]->count();
      caffe_cpu_scale(count, alpha, diff_data, bottom[0]->mutable_cpu_diff());
    }
    // propagate to z hat
    if (propagate_down[1]) {
      alpha = - top[0]->cpu_diff()[0] / bottom[0]->count();
      if (bottom.size()>2) {
      // div the diff by variance
        caffe_div(count, diff_data, tempvar_.cpu_data(), diff_data);
      }   
      caffe_cpu_scale(count, alpha, diff_data, bottom[1]->mutable_cpu_diff());
    }
    // dont propagate to varibles
}

#ifdef CPU_ONLY
STUB_GPU(LadderLossLayer);
#endif

INSTANTIATE_CLASS(LadderLossLayer);
REGISTER_LAYER_CLASS(LadderLoss);

}  // namespace caffe
