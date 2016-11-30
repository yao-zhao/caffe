#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  has_class_prob_ = (bottom.size() == 3) ? true : false;
  lambda_coord_ = this->layer_param_.yolo_loss_param().lambda_coord();
  lambda_noobj_ = this->layer_param_.yolo_loss_param().lambda_noobj();
  eps_ = this->layer_param_.yolo_loss_param().eps();
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // check bounding box prediction
  CHECK_EQ(bottom[0]->num_axes(), 5) <<
      "bounding box prediction bottom[0] must be a five dimension tensor";
  const int batchsize = bottom[0]->shape(0);
  S_h_ = bottom[0]->shape(1);
  S_w_ =  bottom[0]->shape(2);
  B_ = bottom[0]->shape(3);
  CHECK_EQ(bottom[0]->shape(4), 5)
      << "first dimension in bouding box prediction bottom[0] must be 5"
      << " which contains x, y, w, h, c(confidence)";
  // check label
  CHECK_EQ(bottom[1]->num_axes(), 3) <<
      "label bottom[1] must be a three dimension tensor";
  CHECK_EQ(batchsize, bottom[1]->shape(0))
      << "batch size of bottom[0] and bottom[1] have to match";
  maxnumbox_ = bottom[1]->shape(1);
  CHECK_EQ(bottom[1]->shape(2), 5)
      << "third dimension of bounding box label bottom[1] must be 5"
      << " which contains, x, y, w, h, C(class)";
  // grid wise class prediction
  if (has_class_prob_) {
    CHECK_EQ(bottom[2]->num_axes(), 4) <<
        "class prediction bottom[2] must be a 4 dimension tensor";
    CHECK_EQ(batchsize, bottom[2]->shape(0))
          << "batch size of bottom[0] and bottom[2] have to match";
    CHECK_EQ(bottom[2]->shape(1), S_h_)
        << "grid height of bounding box predction bottom[0]"
        << "and class prediction[2] has to match.";
    CHECK_EQ(bottom[2]->shape(2), S_w_)
        << "grid width of bounding box predction bottom[0]"
        << "and class prediction[2] has to match.";
  }
  // saves which bounding box it belongs to, use the closest one
  vector<int> Ii_dim;
  Ii_dim.push_back(batchsize);
  Ii_dim.push_back(S_h_);
  Ii_dim.push_back(S_w_);
  grid_label_.Reshape(Ii_dim);
  Ii_dim.push_back(maxnumbox_);
  Ii_dim.push_back(5);
  diff_.Reshape(Ii_dim);
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batchsize = bottom[0]->shape(0);
  const Dtype* bbox_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype delta_h = 1. / Dtype(S_h_);
  const Dtype delta_w = 1. / Dtype(S_w_);
  Dtype* bbox_diff = diff_.mutable_cpu_data();
  Dtype* grid_label_data = grid_label_.mutable_cpu_data();
  caffe_set(grid_label_.count(), Dtype(0), grid_label_data);
  caffe_set(bottom[0]->count(), Dtype(0), bbox_diff);
  Dtype loss = Dtype(0);
  for (int ibatch = 0; ibatch < batchsize; ++ibatch) {
    for (int ih = 0; ih < S_h_; ++ih) {
      const Dtype center_h = delta_h * (Dtype(ih) + 0.5);
      const Dtype start_h = delta_h * Dtype(ih);
      for (int iw = 0; iw < S_w_; ++iw) {
        const Dtype center_w = delta_w * (Dtype(iw) + 0.5);
        const Dtype start_w = delta_w * Dtype(iw);
        const int Ii_index = (((ibatch * S_h_) + ih) * S_w_ + iw);
        // assign the closest bounding box label to grid
        Dtype choose_label = -1.0;
        Dtype dist2 = Dtype(FLT_MAX);
        for (int ilabel = 0; ilabel < maxnumbox_; ++ilabel) {
          const int label_index = ((ibatch * maxnumbox_) + ilabel) * 5;
          const Dtype label_C = label_data[label_index+4];
          if (label_C >= 0) {
            const Dtype diff_x = label_data[label_index] - center_w;
            const Dtype diff_y = label_data[label_index+1] - center_h;
            // const Dtype label_w = label_data[label_index+2];
            // const Dtype label_h = label_data[label_index+3];
            const Dtype dist2_new = diff_x * diff_x + diff_y * diff_y;
            // only if the CENTER of the label box is in that grid
            if ((fabs(diff_x) < (delta_w / 2.0)) &&
                (fabs(diff_y) < (delta_h / 2.0)) &&
                (dist2_new < dist2)) {
            // if ((fabs(diff_x) < (label_w / 2.0)) &&
            //     (fabs(diff_y) < (label_h / 2.0)) &&
              choose_label = Dtype(ilabel);
              dist2 = dist2_new;
            }
          }
        }
        grid_label_data[Ii_index] = choose_label;
        if (choose_label >= 0) {
          const int label_index = ((ibatch * maxnumbox_) + choose_label) * 5;
          const Dtype label_x = label_data[label_index];
          const Dtype label_y = label_data[label_index+1];
          const Dtype label_w = label_data[label_index+2];
          const Dtype label_h = label_data[label_index+3];
          // loop through bounding box
          Dtype iou = 0.0;
          int j = 0;
          for (int ib = 0; ib < B_; ++ib) {
            const int bbox_index = (Ii_index * B_ + ib) * 5;
            const Dtype bbox_x = start_w + bbox_data[bbox_index] * delta_w;
            const Dtype bbox_y = start_h + bbox_data[bbox_index+1] * delta_h;
            const Dtype bbox_w = bbox_data[bbox_index+2];
            const Dtype bbox_h = bbox_data[bbox_index+3];
            Dtype min_w = (bbox_w + label_w) / Dtype(2)
                - fabs(bbox_x - label_x);
            Dtype min_h = (bbox_h + label_h) / Dtype(2)
                - fabs(bbox_y - label_y);
            min_w = min(min_w, min(bbox_w, label_w));
            min_h = min(min_h, min(bbox_h, label_h));
            Dtype iou_j = ((min_w > 0) && (min_h > 0)) ? (min_w * min_h) /
                (bbox_w * bbox_h + label_w * label_h - min_w * min_h) : 0.0;
            if (iou_j > iou) {
              iou = iou_j;
              j = ib;
            }
          }
          // calculate diff for jth box
          const int Iij_index = Ii_index * B_ + j;
          const int bbox_index = Iij_index * 5;
          const Dtype bbox_x = start_w + bbox_data[bbox_index] * delta_w;
          const Dtype bbox_y = start_h + bbox_data[bbox_index+1] * delta_h;
          const Dtype bbox_w_sqrt = sqrt(bbox_data[bbox_index+2]);
          const Dtype bbox_h_sqrt = sqrt(bbox_data[bbox_index+3]);
          bbox_diff[bbox_index] = lambda_coord_ * (bbox_x - label_x);
          bbox_diff[bbox_index+1] = lambda_coord_ * (bbox_y - label_y);
          bbox_diff[bbox_index+2] =
              lambda_coord_ * (bbox_w_sqrt - sqrt(label_w));
          bbox_diff[bbox_index+3] =
              lambda_coord_ * (bbox_h_sqrt - sqrt(label_h));
          for (int k = 0; k < 4; ++k) {
            loss += bbox_diff[bbox_index + k] *
                bbox_diff[bbox_index + k] / lambda_coord_ / 2.0;
          }
          bbox_diff[bbox_index+2] /= 2*bbox_w_sqrt;
          bbox_diff[bbox_index+3] /= 2*bbox_h_sqrt;
          // calculate diff confidence
          for (int ib = 0; ib < B_; ++ib) {
            const Dtype Ci = (j == ib) ? iou : Dtype(0);
            const int bbox_index_ib = (Ii_index * B_ + ib) * 5 + 4;
            bbox_diff[bbox_index_ib] =
                lambda_coord_ * (bbox_data[bbox_index_ib] - Ci);
            loss += bbox_diff[bbox_index_ib] * bbox_diff[bbox_index_ib] /
                lambda_coord_ / 2.0;
        // // calculate partial iou over dx, dy, dw, dh
        // if (j == ib) {
        //   Dtype dx, dy, dw, dh;
        //   const int bbox_index = (Ii_index * B_ + ib) * 5;
        //   const Dtype bbox_x = bbox_data[bbox_index];
        //   const Dtype bbox_y = bbox_data[bbox_index+1];
        //   const Dtype bbox_w = bbox_data[bbox_index+2];
        //   const Dtype bbox_h = bbox_data[bbox_index+3];
        //   Dtype min_w = (bbox_w + label_w) / Dtype(2)
        //       - fabs(bbox_x - label_x);
        //   Dtype min_h = (bbox_h + label_h) / Dtype(2)
        //       - fabs(bbox_y - label_y);
        //   // dmin_w
        //   if ((bbox_w < label_w) && (bbox_w < min_w)) {
        //     dw = 1.0;
        //     dx = 0.0;
        //     min_w = bbox_w;
        //   } else if ((min_w < bbox_w) && (min_w < label_w)) {
        //     dw = 0.5;
        //     if (bbox_x > label_x) {
        //       dx = -1.0;
        //     } else if (bbox_x < label_x) {
        //       dx = 1.0;
        //     } else {
        //       dx = 0;
        //     }
        //     min_w = min_w;
        //   } else {
        //     dx = 0.0;
        //     dw = 0.0;
        //     min_w = label_w;
        //   }
        //   // dmin_h
        //   if ((bbox_h < label_h) && (bbox_h < min_h)) {
        //     dh = 1.0;
        //     dy = 0.0;
        //     min_h = bbox_h;
        //   } else if ((min_h < bbox_h) && (min_h < label_h)) {
        //     dh = 0.5;
        //     if (bbox_h > label_h) {
        //       dy = -1.0;
        //     } else if (bbox_y < label_y) {
        //       dy = 1.0;
        //     } else {
        //       dy = 0;
        //     }
        //     min_h = min_h;
        //   } else {
        //     dy = 0.0;
        //     dh = 0.0;
        //     min_h = label_h;
        //   }
        //   Dtype iou_j = ((min_w > 0) && (min_h > 0)) ? (min_w * min_h) /
        //       (bbox_w * bbox_h + label_w * label_h - min_w * min_h) : 0.0;
        // }
          }
        } else {
          // calculate diff confidence for none object
          for (int ib = 0; ib < B_; ++ib) {
            const int bbox_index_ib = (Ii_index * B_ + ib) * 5 + 4;
            bbox_diff[bbox_index_ib] =
                lambda_noobj_ * bbox_data[bbox_index_ib];
            loss += bbox_diff[bbox_index_ib] * bbox_diff[bbox_index_ib] /
                lambda_noobj_ / 2.0;
          }
        }
      }
    }
  }
  // if have softmax input
  if (has_class_prob_) {
    int num_classes = bottom[2]->shape(3);
    const Dtype* prob_data = bottom[2]->cpu_data();
    for (int i = 0; i < batchsize * S_h_ * S_w_; ++i) {
      for (int c = 0; c < num_classes; ++c) {
        if (Dtype(c) == grid_label_data[i]) {
          loss += - log(min(prob_data[i * num_classes + c], Dtype(FLT_MIN)));
        }
      }
    }
  }
  // normalize over batch
  loss /= Dtype(batchsize);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), diff_.cpu_data(), bottom_diff);
    Dtype loss_weight = top[0]->cpu_diff()[0] / Dtype(bottom[0]->shape(0));
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
  if (has_class_prob_) {
    if (propagate_down[2]) {
      const int num_classes = bottom[2]->shape(3);
      const int outer_dim = bottom[2]->count(0, 3);
      const Dtype* grid_label_data = grid_label_.cpu_data();
      Dtype* prob_diff = bottom[2]->mutable_cpu_diff();
      for (int i = 0; i < outer_dim; ++i) {
        for (int c = 0; c < num_classes; ++c) {
          if (Dtype(c) == grid_label_data[i]) {
            prob_diff[i * num_classes + c] = -1;
          }
        }
      }
      Dtype loss_weight = top[0]->cpu_diff()[0] / Dtype(bottom[0]->shape(0));
      caffe_scal(bottom[0]->count(), loss_weight, prob_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(YoloLossLayer);
#endif

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);

}  // namespace caffe
