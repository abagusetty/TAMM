#ifndef TAMMX_BLOCK_H_
#define TAMMX_BLOCK_H_

#include <memory>

#include "tammx/errors.h"
#include "tammx/types.h"
#include "tammx/tce.h"

namespace tammx {

template<typename T>
class LabeledBlock;

template<typename T>
class Tensor;

template<typename T>
class Block {
 public:

  Block() = delete;
  Block(const Block<T>&) = delete;
  Block<T>& operator = (const Block<T>&) = delete;
  
  Block(Tensor<T>& tensor,
        const BlockDimVec& block_id)
    : tensor_{tensor},
      block_id_{block_id} {
        block_dims_ = tensor.block_dims(block_id);
        layout_.resize(tensor.rank());
        std::iota(layout_.begin(), layout_.end(), 0);
        sign_ = 1;
        EXPECTS(size() > 0);
        buf_ = std::make_unique<T[]> (size());
      }

  Block(Block<T>&& block)
      : tensor_{block.tensor_},
        block_id_{block.block_id_},
        block_dims_{block.block_dims_},
        layout_{block.layout_},
        sign_{block.sign_},
        buf_{std::move(block.buf_)} { }
  
  Block(Tensor<T>& tensor,
        const BlockDimVec& block_id,
        const PermVec& layout,
        Sign sign)
      : tensor_{tensor},
        block_id_{block_id},
        layout_{layout},
        sign_{sign} {
          block_dims_ = tensor.block_dims(block_id);
          EXPECTS(tensor.rank() == block_id.size());
          EXPECTS(tensor.rank() == block_dims_.size());
          EXPECTS(tensor.rank() == layout.size());
          EXPECTS(size() > 0);
          buf_ = std::make_unique<T[]> (size());
        }

  const BlockDimVec& blockid() const {
    return block_id_;
  }

  BlockDimVec block_offset() const {
    BlockDimVec ret;
    for(auto id: block_id_) {
      ret.push_back(BlockIndex{TCE::offset(id)});
    }
    return ret;
  }
  
  const BlockDimVec& block_dims() const {
    return block_dims_;
  }

  LabeledBlock<T> operator () (const IndexLabelVec& label) {
    return {this, label};
  }

  LabeledBlock<T> operator () () {
    IndexLabelVec label;
    for(int i=0; i<block_id_.size(); i++) {
      label.push_back({i, tensor_.flindices()[i]});
    }
    return operator ()(label);
  }
  
  size_t size() const {
    size_t sz = 1;
    for(auto x : block_dims_) {
      sz *= x.value();
    }
    return sz;
  }

  Sign sign() const {
    return sign_;
  }

  const PermVec& layout() const {
    return layout_;
  }

  T* buf() {
    return buf_.get();
  }

  const T* buf() const {
    return buf_.get();
  }

  Tensor<T>& tensor() {
    return tensor_;
  }
  
 private:
  Tensor<T>& tensor_;
  BlockDimVec block_id_;
  BlockDimVec block_dims_;
  std::unique_ptr<T[]> buf_;
  PermVec layout_;
  Sign sign_;
};


}  // namespace tammx

#endif  // TAMMX_BLOCK_H_

