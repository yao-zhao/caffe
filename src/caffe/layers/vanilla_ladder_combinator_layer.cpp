#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/vanilla_ladder_combinator_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const LadderCombinatorParameter& param = this->layer_param_.ladder_combinator_param();
  // check axis, default starting axis is 1 and default num_axes is -1
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  const int num_axes = param.num_axes();
  CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
  << "or -1 to extend to the end of bottom[0]";
  if (num_axes >= 0) {
    CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
    << "scale blob's shape extends past bottom[0]'s shape when applied "
    << "starting with bottom[0] axis = " << axis_;
  }
  // initiallized 9 blobs
  this->blobs_.resize(9);
  const vector<int>::const_iterator& shape_start =
      bottom[0]->shape().begin() + axis_;
  const vector<int>::const_iterator& shape_end =
      (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
  vector<int> scale_shape(shape_start, shape_end);
  for (int iblob=0; iblob<9; ++iblob) {
    this->blobs_[iblob].reset(new Blob<Dtype>(scale_shape));
  }
  // fill each layer with default
  // 0b matrix
  FillerParameter filler_0b_param(param.filler_0b());
  if (!param.has_filler_0b()) {
    filler_0b_param.set_type("constant");
    filler_0b_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_0b(GetFiller<Dtype>(filler_0b_param));
  filler_0b->Fill(this->blobs_[0].get());
  // 0z matrix
  FillerParameter filler_0z_param(param.filler_0z());
  if (!param.has_filler_0z()) {
    filler_0z_param.set_type("constant");
    filler_0z_param.set_value(1);
  }
  shared_ptr<Filler<Dtype> > filler_0z(GetFiller<Dtype>(filler_0z_param));
  filler_0z->Fill(this->blobs_[1].get());
  // 0u matrix
  FillerParameter filler_0u_param(param.filler_0u());
  if (!param.has_filler_0u()) {
    filler_0u_param.set_type("constant");
    filler_0u_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_0u(GetFiller<Dtype>(filler_0u_param));
  filler_0u->Fill(this->blobs_[2].get());
  // 0zu matrix
  FillerParameter filler_0zu_param(param.filler_0zu());
  if (!param.has_filler_0zu()) {
    filler_0zu_param.set_type("constant");
    filler_0zu_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_0zu(GetFiller<Dtype>(filler_0zu_param));
  filler_0zu->Fill(this->blobs_[3].get());
  // sigma matrix
  FillerParameter filler_sigma_param(param.filler_sigma());
  if (!param.has_filler_sigma()) {
    filler_sigma_param.set_type("constant");
    // filler_sigma_param.set_value(0);
    filler_sigma_param.set_value(1);
  }
  shared_ptr<Filler<Dtype> > filler_sigma(GetFiller<Dtype>(filler_sigma_param));
  filler_sigma->Fill(this->blobs_[4].get());
    // 1b matrix
  FillerParameter filler_1b_param(param.filler_1b());
  if (!param.has_filler_1b()) {
    filler_1b_param.set_type("constant");
    filler_1b_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_1b(GetFiller<Dtype>(filler_1b_param));
  filler_1b->Fill(this->blobs_[5].get());
  // 1z matrix
  FillerParameter filler_1z_param(param.filler_1z());
  if (!param.has_filler_1z()) {
    filler_1z_param.set_type("constant");
    filler_1z_param.set_value(1);
  }
  shared_ptr<Filler<Dtype> > filler_1z(GetFiller<Dtype>(filler_1z_param));
  filler_1z->Fill(this->blobs_[6].get());
  // 1u matrix
  FillerParameter filler_1u_param(param.filler_1u());
  if (!param.has_filler_1u()) {
    filler_1u_param.set_type("constant");
    filler_1u_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_1u(GetFiller<Dtype>(filler_1u_param));
  filler_1u->Fill(this->blobs_[7].get());
  // 1zu matrix
  FillerParameter filler_1zu_param(param.filler_1zu());
  if (!param.has_filler_1zu()) {
    filler_1zu_param.set_type("constant");
    filler_1zu_param.set_value(0);
  }
  shared_ptr<Filler<Dtype> > filler_1zu(GetFiller<Dtype>(filler_1zu_param));
  filler_1zu->Fill(this->blobs_[8].get());
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // check to make sure that the input blobs have the same size
  CHECK(bottom[0]->shape() == bottom[1]->shape());
  top[0]->ReshapeLike(*bottom[0]);
  // set axis information
  for (int i = 0; i < this->blobs_[0]->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), this->blobs_[0]->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  // set the inner and outer dimension
  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_ + 1);
  comb_dim_ = bottom[0]->count(axis_);

  // initialize temp
  tempmul_.ReshapeLike(*bottom[0]);
  tempsig_.ReshapeLike(*bottom[0]);

  // initialize sum multiplier
  sum_multiplier_.Reshape(vector<int>(1, outer_dim_));
  if (sum_multiplier_.cpu_data()[outer_dim_ - 1] != Dtype(1)) {
    caffe_set(outer_dim_, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_z = bottom[0]->cpu_data();
  const Dtype* bottom_data_u = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight_b0 = this->blobs_[0]->cpu_data();
  const Dtype* weight_w0z = this->blobs_[1]->cpu_data();
  const Dtype* weight_w0u = this->blobs_[2]->cpu_data();
  const Dtype* weight_w0zu = this->blobs_[3]->cpu_data();
  const Dtype* weight_wsigma = this->blobs_[4]->cpu_data();
  const Dtype* weight_b1 = this->blobs_[5]->cpu_data();
  const Dtype* weight_w1z = this->blobs_[6]->cpu_data();
  const Dtype* weight_w1u = this->blobs_[7]->cpu_data();
  const Dtype* weight_w1zu = this->blobs_[8]->cpu_data();
  // Dtype* temp_data = temp_.mutable_cpu_data();
  Dtype* tempsig_data = tempsig_.mutable_cpu_data();
  Dtype* tempmul_data = tempmul_.mutable_cpu_data();
  int count = bottom[0]->count();
  int inner_id;
  // currently does not support in-place
  CHECK(bottom[0] != top[0]) <<
    "currently does not support in-place for this ladder layer for bottom 0";
  CHECK(bottom[1] != top[0]) <<
    "currently does not support in-place for this ladder layer for bottom 1";
  for (int i = 0; i < count; ++i) {
    inner_id = i%comb_dim_;
    tempmul_data[i] = bottom_data_z[i] * bottom_data_u[i];
    tempsig_data[i] = sigmoid(weight_b1[inner_id] +
      weight_w1z[inner_id] * bottom_data_z[i] +
      weight_w1u[inner_id] * bottom_data_u[i] +
      weight_w1zu[inner_id] * tempmul_data[i]);
    top_data[i] = tempsig_data[i] * weight_wsigma[inner_id] +
      weight_b0[inner_id] +
      weight_w0z[inner_id] * bottom_data_z[i] +
      weight_w0u[inner_id] * bottom_data_u[i] +
      weight_w0zu[inner_id] * tempmul_data[i];
  }

  // for (int n = 0; n < outer_dim_; ++n) {
  //   // temp mult
  //   caffe_mul<Dtype>(comb_dim_, bottom_data_z, bottom_data_u, tempmul_data);
  //   // copy bias
  //   caffe_copy(comb_dim_, weight_b0, top_data);
  //   caffe_copy(comb_dim_, weight_b1, tempsig_data);
  //   // add to combination
  //   caffe_mul<Dtype>(comb_dim_, weight_w0z, bottom_data_z, temp_data);
  //   caffe_add<Dtype>(comb_dim_, top_data, temp_data, top_data);
  //   caffe_mul<Dtype>(comb_dim_, weight_w0u, bottom_data_u, temp_data);
  //   caffe_add<Dtype>(comb_dim_, top_data, temp_data, top_data);
  //   caffe_mul<Dtype>(comb_dim_, weight_w0zu, tempmul_data, temp_data);
  //   caffe_add<Dtype>(comb_dim_, top_data, temp_data, top_data);
  //   // add to sigmoid
  //   caffe_mul<Dtype>(comb_dim_, weight_w1z, bottom_data_z, tempsig_data);
  //   caffe_add<Dtype>(comb_dim_, tempsig_data, temp_data, tempsig_data);
  //   caffe_mul<Dtype>(comb_dim_, weight_w1u, bottom_data_u, temp_data);
  //   caffe_add<Dtype>(comb_dim_, tempsig_data, temp_data, tempsig_data);
  //   caffe_mul<Dtype>(comb_dim_, weight_w1zu, tempmul_data, temp_data);
  //   caffe_add<Dtype>(comb_dim_, tempsig_data, temp_data, tempsig_data);
  //   // sigmod function
  //   for (int i = 0; i < comb_dim_; ++i) {
  //     tempsig_data[i] = sigmoid(tempsig_data[i]);
  //   }
  //   // final
  //   caffe_mul<Dtype>(comb_dim_, weight_wsigma, tempsig_data, tempsig_data);
  //   caffe_add<Dtype>(comb_dim_, tempsig_data, top_data, top_data);
  //   // iterate
  //   bottom_data_z += comb_dim_;
  //   bottom_data_u += comb_dim_;
  //   top_data += comb_dim_;
  //   tempmul_data += comb_dim_;
  //   tempsig_data += comb_dim_;
  // }
}

template <typename Dtype>
void VanillaLadderCombinatorLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data_z = bottom[0]->cpu_data();
    const Dtype* bottom_data_u = bottom[1]->cpu_data();
    const Dtype* weight_w0z = this->blobs_[1]->cpu_data();
    const Dtype* weight_w0u = this->blobs_[2]->cpu_data();
    const Dtype* weight_w0zu = this->blobs_[3]->cpu_data();
    const Dtype* weight_wsigma = this->blobs_[4]->cpu_data();
    const Dtype* weight_w1z = this->blobs_[6]->cpu_data();
    const Dtype* weight_w1u = this->blobs_[7]->cpu_data();
    const Dtype* weight_w1zu = this->blobs_[8]->cpu_data();
    Dtype* bottom_diff_z = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff_u = bottom[1]->mutable_cpu_diff();
    Dtype* weight_diff_b0 = this->blobs_[0]->mutable_cpu_diff();
    Dtype* weight_diff_w0z = this->blobs_[1]->mutable_cpu_diff();
    Dtype* weight_diff_w0u = this->blobs_[2]->mutable_cpu_diff();
    Dtype* weight_diff_w0zu = this->blobs_[3]->mutable_cpu_diff();
    Dtype* weight_diff_wsigma = this->blobs_[4]->mutable_cpu_diff();
    Dtype* weight_diff_b1 = this->blobs_[5]->mutable_cpu_diff();
    Dtype* weight_diff_w1z = this->blobs_[6]->mutable_cpu_diff();
    Dtype* weight_diff_w1u = this->blobs_[7]->mutable_cpu_diff();
    Dtype* weight_diff_w1zu = this->blobs_[8]->mutable_cpu_diff();
    Dtype* tempsig_data = tempsig_.mutable_cpu_data();
    Dtype* tempmul_data = tempmul_.mutable_cpu_data();
    // set weight to zeros,
    // TO DO: add if clause to make it faster for cases that no prop is needed
    int count = bottom[0]->count();
    int inner_id = 0;
    for (int i = 0; i < count; ++i) {
      inner_id = i%comb_dim_;
      weight_diff_b0[inner_id] += top_diff[i];
      weight_diff_w0z[inner_id] += top_diff[i] * bottom_data_z[i];
      weight_diff_w0u[inner_id] += top_diff[i] * bottom_data_u[i];
      weight_diff_w0zu[inner_id] += top_diff[i] * tempmul_data[i];
      weight_diff_wsigma[inner_id] += top_diff[i] * tempsig_data[i];

    // changed tempsig_data here, attention if it has further use
      // tmp = tempsig_data[i] * weight_wsigma[inner_id] * top_diff[i];
      tempsig_data[i] = tempsig_data[i] * (1-tempsig_data[i]) *
        weight_wsigma[inner_id] * top_diff[i];
      weight_diff_b1[inner_id] += tempsig_data[i];
      weight_diff_w1z[inner_id] += tempsig_data[i] * bottom_data_z[i];
      weight_diff_w1u[inner_id] += tempsig_data[i] * bottom_data_u[i];
      weight_diff_w1zu[inner_id] += tempsig_data[i] * tempmul_data[i];

      if (propagate_down[0]) {
        bottom_diff_z[i] = top_diff[i] * (weight_w0z[inner_id] +
          weight_w0zu[inner_id] * bottom_data_u[i]) +
        tempsig_data[i] * (weight_w1z[inner_id] +
          weight_w1zu[inner_id] * bottom_data_u[i]);
      }
      if (propagate_down[1]) {
        bottom_diff_u[i] = top_diff[i] * (weight_w0u[inner_id] +
          weight_w0zu[inner_id] * bottom_data_z[i]) +
        tempsig_data[i] * (weight_w1u[inner_id] +
          weight_w1zu[inner_id] * bottom_data_z[i]);
      }
    }

// for (int n = 0; n < outer_dim_; ++n) {
//   if (this->param_propagate_down_[0]) {
//   // add diff to b0
//     caffe_add<Dtype>(comb_dim_, top_diff, weight_diff_b0, weight_diff_b0);
//   }
//   if (this->param_propagate_down_[1]) {
//   // add diff to w0z
//     caffe_mul<Dtype>(comb_dim_, bottom_data_z, top_diff, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data,
//        weight_diff_w0z, weight_diff_w0z);
//   }
//   if (this->param_propagate_down_[2]) {
//   // add diff to w0u
//     caffe_mul<Dtype>(comb_dim_, bottom_data_u, top_diff, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data, weight_diff_w0u, weight_diff_w0u);
//   }
//   if (this->param_propagate_down_[3]) {
//   // add diff to w0zu
//     caffe_mul<Dtype>(comb_dim_, tempmul_data, top_diff, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data,
//        weight_diff_w0zu, weight_diff_w0zu);
//   }
//   if (this->param_propagate_down_[4]) {
//   // add diff to wsigma
//     caffe_mul<Dtype>(comb_dim_, top_diff, tempsig_data, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data,
//       weight_diff_wsigma, weight_diff_wsigma);
//   }
//   // store sigmoid diff in tempsig_data S*(1-S)*wsigma*Z,
//   // override tempsig_data
//   // caution: override tempsig_data, dont use it afterwards
//   for (int i = 0; i < comb_dim_; ++i) {
//     tempsig_data[i] = (1- tempsig_data[i]) * tempsig_data[i];
//   }
//   if (this->param_propagate_down_[5]) {
//     caffe_mul<Dtype>(comb_dim_, weight_wsigma, tempsig_data, tempsig_data);
//     caffe_mul<Dtype>(comb_dim_, top_diff, tempsig_data, tempsig_data);
//   }
//   if (this->param_propagate_down_[6]) {
//   // add diff to b1
//     caffe_add<Dtype>(comb_dim_, tempsig_data,
//       weight_diff_b1, weight_diff_b1);
//   }
//   if (this->param_propagate_down_[7]) {
//   // diff w1z
//     caffe_mul<Dtype>(comb_dim_, tempsig_data, bottom_data_z, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data, weight_diff_w1z, weight_diff_w1z);
//   }
//   if (this->param_propagate_down_[8]) {
//   // diff w1u
//     caffe_mul<Dtype>(comb_dim_, tempsig_data, bottom_data_u, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data, weight_diff_w1u, weight_diff_w1u);
//   }
//   if (this->param_propagate_down_[9]) {
//   // diff w1zu
//     caffe_mul<Dtype>(comb_dim_, tempsig_data, tempmul_data, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data,
//       weight_diff_w1zu, weight_diff_w1zu);
//   }
//   if (propagate_down[0]) {
//   // calculate bottom diff z
//     caffe_mul<Dtype>(comb_dim_, weight_w1zu, bottom_data_u, bottom_diff_z);
//     caffe_add<Dtype>(comb_dim_, weight_w1z, bottom_diff_z, bottom_diff_z);
//     caffe_mul<Dtype>(comb_dim_, tempsig_data, bottom_diff_z, bottom_diff_z);
//     caffe_mul<Dtype>(comb_dim_, weight_w0zu, bottom_data_u, temp_data);
//     caffe_add<Dtype>(comb_dim_, weight_w0z, temp_data, temp_data);
//     caffe_mul<Dtype>(comb_dim_, top_diff, temp_data, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data, bottom_diff_z, bottom_diff_z);
//   }
//   if (propagate_down[1]) {
//   // calculate bottom diff u
//     caffe_mul<Dtype>(comb_dim_, weight_w1zu, bottom_data_z, bottom_diff_u);
//     caffe_add<Dtype>(comb_dim_, weight_w1u, bottom_diff_u, bottom_diff_u);
//     caffe_mul<Dtype>(comb_dim_, tempsig_data, bottom_diff_u, bottom_diff_u);
//     caffe_mul<Dtype>(comb_dim_, weight_w0zu, bottom_data_z, temp_data);
//     caffe_add<Dtype>(comb_dim_, weight_w0u, temp_data, temp_data);
//     caffe_mul<Dtype>(comb_dim_, top_diff, temp_data, temp_data);
//     caffe_add<Dtype>(comb_dim_, temp_data, bottom_diff_u, bottom_diff_u);
//   }
//   // iterate
//   bottom_data_z += comb_dim_;
//   bottom_data_u += comb_dim_;
//   bottom_diff_z += comb_dim_;
//   bottom_diff_u += comb_dim_;
//   top_diff += comb_dim_;
//   tempmul_data += comb_dim_;
//   tempsig_data += comb_dim_;
// }
  }

#ifdef CPU_ONLY
STUB_GPU(VanillaLadderCombinatorLayer);
#endif

INSTANTIATE_CLASS(VanillaLadderCombinatorLayer);
REGISTER_LAYER_CLASS(VanillaLadderCombinator);

}  // namespace caffe
