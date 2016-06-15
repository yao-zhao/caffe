#include <cfloat>
#include <vector>

#include "caffe/layers/vanilla_ladder_combinator_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

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
  this->blobs_.resize(8);
  const vector<int>::const_iterator& shape_start =
      bottom[0]->shape().begin() + axis_;
  const vector<int>::const_iterator& shape_end =
      (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
  vector<int> scale_shape(shape_start, shape_end);
  for (int iblob=0; iblob<8; iblob++) {
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
  // 0u matrix
  FillerParameter filler_0u_param(param.filler_0u());
  if (!param.has_filler_0u()) {
    filler_0u_param.set_type("constant");
    filler_0u_param.set_value(0);  
  }
  shared_ptr<Filler<Dtype> > filler_0u(GetFiller<Dtype>(filler_0u_param));
  filler_0u->Fill(this->blobs_[1].get());
  // 0z matrix
  FillerParameter filler_0z_param(param.filler_0z());
  if (!param.has_filler_0z()) {
    filler_0z_param.set_type("constant");
    filler_0z_param.set_value(1);  
  }
  shared_ptr<Filler<Dtype> > filler_0z(GetFiller<Dtype>(filler_0z_param));
  filler_0z->Fill(this->blobs_[2].get());
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
  // 1u matrix
  FillerParameter filler_1u_param(param.filler_1u());
  if (!param.has_filler_1u()) {
    filler_1u_param.set_type("constant");
    filler_1u_param.set_value(0);  
  }
  shared_ptr<Filler<Dtype> > filler_1u(GetFiller<Dtype>(filler_1u_param));
  filler_1u->Fill(this->blobs_[6].get());
  // 1z matrix
  FillerParameter filler_1z_param(param.filler_1z());
  if (!param.has_filler_1z()) {
    filler_1z_param.set_type("constant");
    filler_1z_param.set_value(1);  
  }
  shared_ptr<Filler<Dtype> > filler_1z(GetFiller<Dtype>(filler_1z_param));
  filler_1z->Fill(this->blobs_[7].get());
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
void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // check if the bottom layers have the same dimension
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1)) {
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);
REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
