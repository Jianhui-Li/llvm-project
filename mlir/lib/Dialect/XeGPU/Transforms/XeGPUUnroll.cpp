//===- XeGPUUnroll.cpp - patterns to do unrolling ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for unrolling XeGPU operations. It follows a
// similar concept and design as vector unroll patterns, serving as a complement
// to them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUUNROLL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-unroll"

using namespace mlir;

namespace {

template <typename SourceOp>
struct UnrollPattern : public OpRewritePattern<SourceOp> {
  UnrollPattern(MLIRContext *context, const xegpu::UnrollOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit), options(options) {}

protected:
  /// Return the target shape for the given `op`. Return std::nullopt if the
  /// op shouldn't be or cannot be unrolled.
  std::optional<SmallVector<int64_t>> getTargetShape(Operation *op) const {
    LDBG() << "Get unroll shape for: " << *op;

    if (options.filterConstraint && failed(options.filterConstraint(op))) {
      LDBG() << "--no filter constraint -> BAIL";
      return std::nullopt;
    }

    assert(options.nativeShape &&
           "expects the native shape for native shape call back function.");
    auto nativeShape = options.nativeShape(op);
    return nativeShape;
  }

  SmallVector<Type> getUnrolledTypes(ShapedType type,
                                     ArrayRef<int64_t> tileShape,
                                     bool returnSingleType = false) const {
    return options.getUnrolledTypes(type, tileShape, returnSingleType);
  }

  /// Emulate the the unpack behavior using insert_strided_slice for VectorType
  /// values and unrealized_conversion_cast for TensorDescType values.
  Value unpack(ValueRange srcs, Type destTy, ArrayRef<int64_t> blockSize,
               Location loc, PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(destTy)) {
      auto shape = vecTy.getShape();
      return xegpu::createVectorWithShapeFromValues(rewriter, loc, srcs, shape);
    }

    if (isa<xegpu::TensorDescType>(destTy)) {
      auto attr = NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = UnrealizedConversionCastOp::create(
          rewriter, loc, destTy, srcs,
          ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResult(0);
    }

    llvm_unreachable("Unexpected destTy.");
    return Value();
  }

  /// Emulate the the pack behavior using extract_strided_slice for VectorType
  /// values and unrealized_conversion_cast for TensorDescType values.
  SmallVector<Value> pack(Value src, TypeRange destTypes,
                          ArrayRef<int64_t> blockSize, Location loc,
                          PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(src.getType())) {
      return xegpu::extractVectorsWithShapeFromValue(rewriter, loc, src,
                                                     blockSize);
    }

    if (isa<xegpu::TensorDescType>(src.getType())) {
      auto attr = NamedAttribute(rewriter.getStringAttr(packAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = UnrealizedConversionCastOp::create(
          rewriter, loc, destTypes, src,
          ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResults();
    }

    llvm_unreachable("Unexpected src type.");
    return SmallVector<Value>();
  }

  /// Helper to pack operands for DPAS-like operations with early return if
  /// no unrolling is needed.
  SmallVector<Value> packOperandForDpas(Value operand,
                                        ArrayRef<int64_t> blockSize,
                                        Location loc,
                                        PatternRewriter &rewriter) const {
    auto vecType = cast<VectorType>(operand.getType());
    std::optional<SmallVector<int64_t>> grids =
        computeShapeRatio(vecType.getShape(), blockSize);
    assert(grids && "Expecting grids to be computed.");
    auto numNewOps = computeProduct(*grids);
    if (numNewOps == 1)
      return SmallVector<Value>({operand});
    VectorType newVecTy =
        vecType.cloneWith(blockSize, vecType.getElementType());
    SmallVector<Type> convertedTypes(numNewOps, newVecTy);
    return pack(operand, convertedTypes, blockSize, loc, rewriter);
  }

private:
  const char *const packAttrName = "__xegpu_blocking_pack__";
  const char *const unpackAttrName = "__xegpu_blocking_unpack__";
  const char *const blockAttrName = "__xegpu_blocking_tile_shape__";

  xegpu::UnrollOptions options;
};

// Generic helper function for unrolling operations with offsets.
//
// Iterates over tile offsets within the tensor descriptor shape and calls
// the provided createOp function for each computed offset. This is used by
// operations like LoadNd, StoreNd, CreateNdDesc, and PrefetchNd when they
// have explicit offsets that need to be adjusted for each unrolled tile.
SmallVector<Value> computeUnrolledOffsets(
    SmallVector<OpFoldResult> mixedOffsets, xegpu::TensorDescType tdescTy,
    ArrayRef<int64_t> targetShape,
    const std::function<Value(SmallVector<OpFoldResult>)> &createOp,
    Location loc, PatternRewriter &rewriter) {
  int64_t rank = tdescTy.getRank();
  ArrayRef<int64_t> shape = tdescTy.getShape();

  auto addi = [&](OpFoldResult a, int64_t b) -> Value {
    std::optional<int64_t> maybeInt = getConstantIntValue(a);
    if (maybeInt) {
      return arith::ConstantIndexOp::create(rewriter, loc, *maybeInt + b);
    } else {
      auto aV = llvm::cast<Value>(a);
      auto bV = arith::ConstantIndexOp::create(rewriter, loc, b);
      return rewriter.createOrFold<arith::AddIOp>(loc, aV, bV);
    }
  };

  SmallVector<OpFoldResult> oldOffsets = llvm::to_vector(
      llvm::drop_begin(mixedOffsets, mixedOffsets.size() - rank));
  auto validIdxes =
      llvm::seq<int64_t>(mixedOffsets.size() - rank, mixedOffsets.size());

  SmallVector<Value> newOps;
  for (SmallVector<int64_t> offsets :
       StaticTileOffsetRange(shape, targetShape)) {

    for (auto [idx, oldOff, offset] :
         llvm::zip(validIdxes, oldOffsets, offsets))
      mixedOffsets[idx] = addi(oldOff, offset);

    auto newOp = createOp(mixedOffsets);
    newOps.push_back(newOp);
  }
  return newOps;
}

struct UnrollCreateNdOp : public UnrollPattern<xegpu::CreateNdDescOp> {
  using UnrollPattern<xegpu::CreateNdDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    int64_t rank = tdescTy.getRank();
    int64_t batchRank = rank - 2;

    // For rank <= 2 or non-memref source: existing single-tdesc behavior.
    if (batchRank <= 0 || !isa<MemRefType>(op.getSourceType())) {
      SmallVector<Value> newOps;
      auto newTdescTy = getUnrolledTypes(tdescTy, *targetShape)[0];
      auto newOp = xegpu::CreateNdDescOp::create(
          rewriter, loc, newTdescTy, op.getSource(), op.getMixedSizes(),
          op.getMixedStrides());
      newOps.push_back(newOp);
      Value castOp = unpack(newOps, tdescTy, *targetShape, loc, rewriter);
      rewriter.replaceOp(op, castOp);
      return success();
    }

    // For rank > 2 with memref source: create one tdesc per batch tile via
    // memref.subview. Each subview slices the batch dimensions, so the
    // resulting tdesc has the batch offset baked into its base pointer.
    ArrayRef<int64_t> shape = tdescTy.getShape();
    SmallVector<int64_t> batchShape(shape.begin(), shape.begin() + batchRank);
    SmallVector<int64_t> batchTarget(targetShape->begin(),
                                     targetShape->begin() + batchRank);
    // batchBlockSize = [batchTarget..., innerShape...] — one batch slice.
    SmallVector<int64_t> batchBlockSize(batchTarget);
    batchBlockSize.append(shape.begin() + batchRank, shape.end());

    auto newTdescTy =
        cast<xegpu::TensorDescType>(getUnrolledTypes(tdescTy, batchBlockSize)[0]);

    SmallVector<Value> newOps;
    for (SmallVector<int64_t> batchOffsets :
         StaticTileOffsetRange(batchShape, batchTarget)) {
      // Build subview offsets: [batchOffset0, ..., 0, 0]
      SmallVector<OpFoldResult> svOffsets;
      for (int64_t i = 0; i < batchRank; ++i)
        svOffsets.push_back(rewriter.getIndexAttr(batchOffsets[i]));
      for (int64_t i = 0; i < 2; ++i)
        svOffsets.push_back(rewriter.getIndexAttr(0));

      // Build subview sizes matching batchBlockSize.
      SmallVector<OpFoldResult> svSizes;
      for (int64_t d : batchBlockSize)
        svSizes.push_back(rewriter.getIndexAttr(d));

      // Strides all 1.
      SmallVector<OpFoldResult> svStrides(rank, rewriter.getIndexAttr(1));

      auto subview = memref::SubViewOp::create(rewriter, loc, op.getSource(),
                                               svOffsets, svSizes, svStrides);
      auto newOp = xegpu::CreateNdDescOp::create(
          rewriter, loc, newTdescTy, subview.getResult(),
          SmallVector<OpFoldResult>(), SmallVector<OpFoldResult>());
      newOps.push_back(newOp);
    }

    Value castOp = unpack(newOps, tdescTy, batchBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollPrefetchNdOp : public UnrollPattern<xegpu::PrefetchNdOp> {
  using UnrollPattern<xegpu::PrefetchNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    int64_t rank = tdescTy.getRank();
    int64_t batchRank = rank - 2;

    if (batchRank <= 0) {
      SmallVector<Type> convertedTdescTypes =
          getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);
      SmallVector<Value> convertedTdesc = pack(
          op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

      auto createPrefetch = [&](SmallVector<OpFoldResult> offsets) -> Value {
        xegpu::PrefetchNdOp::create(rewriter, loc, convertedTdesc[0], offsets,
                                    op.getL1HintAttr(), op.getL2HintAttr(),
                                    op.getL3HintAttr(), layout);
        return nullptr;
      };
      computeUnrolledOffsets(op.getMixedOffsets(), tdescTy, *targetShape,
                             createPrefetch, loc, rewriter);
    } else {
      ArrayRef<int64_t> shape = tdescTy.getShape();
      SmallVector<int64_t> batchTarget(targetShape->begin(),
                                       targetShape->begin() + batchRank);
      SmallVector<int64_t> innerShape(shape.begin() + batchRank, shape.end());
      SmallVector<int64_t> innerTarget(targetShape->begin() + batchRank,
                                       targetShape->end());
      SmallVector<int64_t> batchBlockSize(batchTarget);
      batchBlockSize.append(innerShape.begin(), innerShape.end());

      SmallVector<Type> batchTdescTypes =
          getUnrolledTypes(tdescTy, batchBlockSize, /*returnSingleType*/ true);
      SmallVector<Value> batchTdescs = pack(
          op.getTensorDesc(), batchTdescTypes, batchBlockSize, loc, rewriter);

      Type innerElemTy = tdescTy.getElementType();
      auto innerTdescTy = xegpu::TensorDescType::get(
          tdescTy.getContext(), innerShape, innerElemTy, tdescTy.getEncoding(),
          /*layout=*/nullptr);

      SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
      SmallVector<OpFoldResult> innerOffsets(mixedOffsets.begin() + batchRank,
                                            mixedOffsets.end());

      for (auto batchTdesc : batchTdescs) {
        auto createPrefetch = [&](SmallVector<OpFoldResult> offsets) -> Value {
          SmallVector<OpFoldResult> fullOffsets(batchRank,
                                               rewriter.getIndexAttr(0));
          fullOffsets.append(offsets.begin(), offsets.end());
          xegpu::PrefetchNdOp::create(rewriter, loc, batchTdesc, fullOffsets,
                                      op.getL1HintAttr(), op.getL2HintAttr(),
                                      op.getL3HintAttr(), layout);
          return nullptr;
        };
        computeUnrolledOffsets(innerOffsets, innerTdescTy, innerTarget,
                               createPrefetch, loc, rewriter);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollLoadNdOp : public UnrollPattern<xegpu::LoadNdOp> {
  using UnrollPattern<xegpu::LoadNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadNdOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType valueTy = op.getType();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    Type elemTy = tdescTy.getElementType();
    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    int64_t rank = tdescTy.getRank();
    int64_t batchRank = rank - 2;
    SmallVector<Value> newOps;

    if (batchRank <= 0) {
      // Rank <= 2: original behavior with single tdesc.
      SmallVector<Type> convertedTdescTypes =
          getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);
      SmallVector<Value> convertedTdescs = pack(
          op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

      auto createLoad = [&](SmallVector<OpFoldResult> offsets) {
        return xegpu::LoadNdOp::create(
            rewriter, loc, newValueTy, convertedTdescs[0], offsets,
            op.getPackedAttr(), op.getTransposeAttr(), op.getL1HintAttr(),
            op.getL2HintAttr(), op.getL3HintAttr(), layout);
      };
      newOps = computeUnrolledOffsets(op.getMixedOffsets(), tdescTy,
                                      *targetShape, createLoad, loc, rewriter);
    } else {
      // Rank > 2: use per-batch tdescs. Pack using batchBlockSize to match
      // the UnrollCreateNdOp's unpack block size.
      ArrayRef<int64_t> shape = tdescTy.getShape();
      SmallVector<int64_t> batchTarget(targetShape->begin(),
                                       targetShape->begin() + batchRank);
      SmallVector<int64_t> innerShape(shape.begin() + batchRank, shape.end());
      SmallVector<int64_t> innerTarget(targetShape->begin() + batchRank,
                                       targetShape->end());
      SmallVector<int64_t> batchBlockSize(batchTarget);
      batchBlockSize.append(innerShape.begin(), innerShape.end());

      SmallVector<Type> batchTdescTypes =
          getUnrolledTypes(tdescTy, batchBlockSize, /*returnSingleType*/ true);
      SmallVector<Value> batchTdescs = pack(
          op.getTensorDesc(), batchTdescTypes, batchBlockSize, loc, rewriter);

      // Create an inner-only tdesc type for computing inner 2D offsets.
      auto innerTdescTy = xegpu::TensorDescType::get(
          tdescTy.getContext(), innerShape, elemTy, tdescTy.getEncoding(),
          /*layout=*/nullptr);

      // Extract inner offsets from the original mixed offsets.
      SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
      SmallVector<OpFoldResult> innerOffsets(mixedOffsets.begin() + batchRank,
                                            mixedOffsets.end());

      for (auto batchTdesc : batchTdescs) {
        auto createLoad = [&](SmallVector<OpFoldResult> offsets) {
          // Prepend zero offsets for batch dims.
          SmallVector<OpFoldResult> fullOffsets(batchRank,
                                               rewriter.getIndexAttr(0));
          fullOffsets.append(offsets.begin(), offsets.end());
          return xegpu::LoadNdOp::create(
              rewriter, loc, newValueTy, batchTdesc, fullOffsets,
              op.getPackedAttr(), op.getTransposeAttr(), op.getL1HintAttr(),
              op.getL2HintAttr(), op.getL3HintAttr(), layout);
        };
        auto batchLoads = computeUnrolledOffsets(
            innerOffsets, innerTdescTy, innerTarget, createLoad, loc, rewriter);
        newOps.append(batchLoads.begin(), batchLoads.end());
      }
    }

    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreNdOp : public UnrollPattern<xegpu::StoreNdOp> {
  using UnrollPattern<xegpu::StoreNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = op.getValueType();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);

    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);

    int64_t rank = tdescTy.getRank();
    int64_t batchRank = rank - 2;
    size_t valueIndex = 0;

    if (batchRank <= 0) {
      SmallVector<Type> convertedTdescTypes =
          getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);
      SmallVector<Value> convertedTdescs = pack(
          op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

      auto createStore = [&](SmallVector<OpFoldResult> offsets) {
        xegpu::StoreNdOp::create(rewriter, loc, convertedValues[valueIndex++],
                                 convertedTdescs[0], offsets,
                                 op.getL1HintAttr(), op.getL2HintAttr(),
                                 op.getL3HintAttr(), layout);
        return (Value) nullptr;
      };
      computeUnrolledOffsets(op.getMixedOffsets(), tdescTy, *targetShape,
                             createStore, loc, rewriter);
    } else {
      ArrayRef<int64_t> shape = tdescTy.getShape();
      SmallVector<int64_t> batchTarget(targetShape->begin(),
                                       targetShape->begin() + batchRank);
      SmallVector<int64_t> innerShape(shape.begin() + batchRank, shape.end());
      SmallVector<int64_t> innerTarget(targetShape->begin() + batchRank,
                                       targetShape->end());
      SmallVector<int64_t> batchBlockSize(batchTarget);
      batchBlockSize.append(innerShape.begin(), innerShape.end());

      SmallVector<Type> batchTdescTypes =
          getUnrolledTypes(tdescTy, batchBlockSize, /*returnSingleType*/ true);
      SmallVector<Value> batchTdescs = pack(
          op.getTensorDesc(), batchTdescTypes, batchBlockSize, loc, rewriter);

      Type innerElemTy = tdescTy.getElementType();
      auto innerTdescTy = xegpu::TensorDescType::get(
          tdescTy.getContext(), innerShape, innerElemTy, tdescTy.getEncoding(),
          /*layout=*/nullptr);

      SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
      SmallVector<OpFoldResult> innerOffsets(mixedOffsets.begin() + batchRank,
                                            mixedOffsets.end());

      for (auto batchTdesc : batchTdescs) {
        auto createStore = [&](SmallVector<OpFoldResult> offsets) {
          SmallVector<OpFoldResult> fullOffsets(batchRank,
                                               rewriter.getIndexAttr(0));
          fullOffsets.append(offsets.begin(), offsets.end());
          xegpu::StoreNdOp::create(rewriter, loc,
                                   convertedValues[valueIndex++], batchTdesc,
                                   fullOffsets, op.getL1HintAttr(),
                                   op.getL2HintAttr(), op.getL3HintAttr(),
                                   layout);
          return (Value) nullptr;
        };
        computeUnrolledOffsets(innerOffsets, innerTdescTy, innerTarget,
                               createStore, loc, rewriter);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollDpasOp : public UnrollPattern<xegpu::DpasOp> {
  using UnrollPattern<xegpu::DpasOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() < 3)
      return failure();

    // targetShape is [batch..., M, K, N]
    int64_t tsRank = targetShape->size();
    auto M = (*targetShape)[tsRank - 3];
    auto K = (*targetShape)[tsRank - 2];
    auto N = (*targetShape)[tsRank - 1];
    ArrayRef<int64_t> batchDims(targetShape->data(), tsRank - 3);

    // Build block sizes including batch dimensions.
    SmallVector<int64_t> aBlockSize(batchDims);
    aBlockSize.push_back(M);
    aBlockSize.push_back(K);
    SmallVector<int64_t> bBlockSize(batchDims);
    bBlockSize.push_back(K);
    bBlockSize.push_back(N);
    SmallVector<int64_t> cBlockSize(batchDims);
    cBlockSize.push_back(M);
    cBlockSize.push_back(N);

    auto a = op.getLhs();
    auto b = op.getRhs();
    auto c = op.getAcc();

    SmallVector<Value> aVals = packOperandForDpas(a, aBlockSize, loc, rewriter);
    SmallVector<Value> bVals = packOperandForDpas(b, bBlockSize, loc, rewriter);
    SmallVector<Value> cVals;
    if (c)
      cVals = packOperandForDpas(c, cBlockSize, loc, rewriter);

    auto ranges = c ? SmallVector<ValueRange>({aVals, bVals, cVals})
                    : SmallVector<ValueRange>({aVals, bVals});
    if (llvm::any_of(ranges, [](auto &v) { return v.size() == 0; }) ||
        llvm::all_of(ranges, [](auto &v) { return v.size() == 1; }))
      return failure();

    VectorType resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();

    // Compute iteration counts. Batch dims only iterate over M and N (not
    // K-reduction), so compute batch iterations from the C block size.
    int64_t aRank = aShape.size();
    int64_t batchRank = batchDims.size();
    int64_t mIters = aShape[batchRank] / M;
    int64_t kIters = aShape[batchRank + 1] / K;
    int64_t nIters = bShape[batchRank + 1] / N;

    // Compute batch iterations (product of batch dim ratios).
    int64_t batchIters = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchIters *= aShape[d] / batchDims[d];

    SmallVector<Value> newOps;
    for (int64_t batch = 0; batch < batchIters; ++batch) {
      for (int64_t i = 0; i < mIters; ++i) {
        for (int64_t j = 0; j < nIters; ++j) {
          Value tmpC;
          if (c)
            tmpC = cVals[batch * (mIters * nIters) + i * nIters + j];

          for (int64_t k = 0; k < kIters; ++k) {
            Value aVec =
                aVals[batch * (mIters * kIters) + i * kIters + k];
            Value bVec =
                bVals[batch * (kIters * nIters) + k * nIters + j];
            SmallVector<Value> operands({aVec, bVec});
            if (tmpC)
              operands.push_back(tmpC);

            tmpC = xegpu::DpasOp::create(
                rewriter, loc, vecTy, operands,
                xegpu::dropInstDataOnAttrs(op->getAttrs()));
          }
          newOps.push_back(tmpC);
        }
      }
    }
    Value castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollDpasMxOp : public UnrollPattern<xegpu::DpasMxOp> {
  using UnrollPattern<xegpu::DpasMxOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasMxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() < 4)
      return failure();

    // targetShape is [batch..., M, K, N, S]
    int64_t tsRank = targetShape->size();
    auto M = (*targetShape)[tsRank - 4];
    auto K = (*targetShape)[tsRank - 3];
    auto N = (*targetShape)[tsRank - 2];
    auto S = (*targetShape)[tsRank - 1];
    ArrayRef<int64_t> batchDims(targetShape->data(), tsRank - 4);

    SmallVector<int64_t> aBlockSize(batchDims);
    aBlockSize.push_back(M);
    aBlockSize.push_back(K);
    SmallVector<int64_t> bBlockSize(batchDims);
    bBlockSize.push_back(K);
    bBlockSize.push_back(N);
    SmallVector<int64_t> cBlockSize(batchDims);
    cBlockSize.push_back(M);
    cBlockSize.push_back(N);
    SmallVector<int64_t> aScaleBlockSize(batchDims);
    aScaleBlockSize.push_back(M);
    aScaleBlockSize.push_back(S);
    SmallVector<int64_t> bScaleBlockSize(batchDims);
    bScaleBlockSize.push_back(S);
    bScaleBlockSize.push_back(N);

    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getAcc();
    auto ascale = dyn_cast<TypedValue<VectorType>>(op.getScaleA());
    auto bscale = dyn_cast<TypedValue<VectorType>>(op.getScaleB());

    SmallVector<Value> aVals = packOperandForDpas(a, aBlockSize, loc, rewriter);
    SmallVector<Value> bVals = packOperandForDpas(b, bBlockSize, loc, rewriter);
    SmallVector<Value> cVals;
    if (c)
      cVals = packOperandForDpas(c, cBlockSize, loc, rewriter);
    SmallVector<Value> aScaleVals;
    if (ascale)
      aScaleVals = packOperandForDpas(ascale, aScaleBlockSize, loc, rewriter);
    SmallVector<Value> bScaleVals;
    if (bscale)
      bScaleVals = packOperandForDpas(bscale, bScaleBlockSize, loc, rewriter);

    VectorType resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();
    int64_t batchRank = batchDims.size();
    int64_t mIters = aShape[batchRank] / M;
    int64_t kIters = aShape[batchRank + 1] / K;
    int64_t nIters = bShape[batchRank + 1] / N;

    int64_t batchIters = 1;
    for (int64_t d = 0; d < batchRank; ++d)
      batchIters *= aShape[d] / batchDims[d];

    SmallVector<Value> newOps;
    xegpu::DpasMxOp newDpasMxOp;
    for (int64_t batch = 0; batch < batchIters; ++batch) {
      for (int64_t i = 0; i < mIters; ++i) {
        for (int64_t j = 0; j < nIters; ++j) {
          Value tmpC;
          if (c)
            tmpC = cVals[batch * (mIters * nIters) + i * nIters + j];

          for (int64_t k = 0; k < kIters; ++k) {
            Value aVec =
                aVals[batch * (mIters * kIters) + i * kIters + k];
            Value bVec =
                bVals[batch * (kIters * nIters) + k * nIters + j];
            SmallVector<Value> operands({aVec, bVec});
            if (tmpC)
              operands.push_back(tmpC);
            if (ascale)
              operands.push_back(
                  aScaleVals[batch * (mIters * kIters) + i * kIters + k]);
            if (bscale)
              operands.push_back(
                  bScaleVals[batch * (kIters * nIters) + k * nIters + j]);

            newDpasMxOp = xegpu::DpasMxOp::create(
                rewriter, loc, vecTy, operands,
                xegpu::dropInstDataOnAttrs(op->getAttrs()));
            tmpC = newDpasMxOp.getResult();
          }
          newOps.push_back(newDpasMxOp);
        }
      }
    }
    Value castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// This pattern handles the unrolling of LoadGatherOp with offsets (gathered
/// load).
/// It unrolls the offsets and mask operands accordingly, and creates multiple
/// LoadGatherOp with the unrolled operands.
struct UnrollLoadGatherOp : public UnrollPattern<xegpu::LoadGatherOp> {
  using UnrollPattern<xegpu::LoadGatherOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    Value offsets = op.getOffsets();
    Value mask = op.getMask();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<int64_t> targetMaskShape(*targetShape);
    int64_t chunkSize = 1;
    if (auto chunkSizeAttr = op->getAttr("chunk_size")) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(chunkSizeAttr))
        chunkSize = intAttr.getInt();
    }

    // Unroll mask and offsets with correct shape
    VectorType maskTy = llvm::dyn_cast<VectorType>(mask.getType());
    VectorType offsetsTy = llvm::dyn_cast<VectorType>(offsets.getType());
    Type elemTy = valueTy.getElementType();
    VectorType newValueTy = VectorType::get(*targetShape, elemTy);

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;
    SmallVector<Type> convertedOffsetTypes;
    SmallVector<Value> convertedOffsets;

    if (chunkSize > 1) {
      // For chunked loads, mask and offsets have one less dimension
      targetMaskShape.pop_back();
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = chunkSize / blockedChunkSize;
      chunkSize = blockedChunkSize;

      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedOffsetTypes = getUnrolledTypes(offsetsTy, targetMaskShape);

      SmallVector<Value> convertedMasksBase =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);
      SmallVector<Value> convertedOffsetsBase =
          pack(offsets, convertedOffsetTypes, targetMaskShape, loc, rewriter);

      for (auto maskVal : convertedMasksBase)
        convertedMasks.append(numNewChunks, maskVal);

      for (auto [baseOffset, offsetType] :
           llvm::zip(convertedOffsetsBase, convertedOffsetTypes)) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          Value inc = arith::ConstantIndexOp::create(rewriter, loc,
                                                     i * blockedChunkSize);
          Value incVec =
              vector::BroadcastOp::create(rewriter, loc, offsetType, inc);
          Value offsetVal =
              arith::AddIOp::create(rewriter, loc, baseOffset, incVec);
          convertedOffsets.push_back(offsetVal);
        }
      }
    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedMasks =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);

      convertedOffsetTypes = getUnrolledTypes(offsetsTy, *targetShape);
      convertedOffsets =
          pack(offsets, convertedOffsetTypes, *targetShape, loc, rewriter);
    }

    auto layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Value> newOps;
    for (auto [o, m] : llvm::zip(convertedOffsets, convertedMasks)) {
      auto newOp = xegpu::LoadGatherOp::create(
          rewriter, loc, newValueTy, op.getSource(), o, m,
          rewriter.getI64IntegerAttr(chunkSize), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr(), layout);
      newOps.push_back(newOp);
    }

    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// This pattern handles the unrolling of StoreScatterOp with offsets (scattered
/// store).
/// It unrolls the offsets and mask operands accordingly, and creates multiple
/// StoreScatterOp with the unrolled operands.
struct UnrollStoreScatterOp : public UnrollPattern<xegpu::StoreScatterOp> {
  using UnrollPattern<xegpu::StoreScatterOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getValue().getType());
    Value offsets = op.getOffsets();
    Value mask = op.getMask();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    int64_t chunkSize = 1;
    if (auto chunkSizeAttr = op->getAttr("chunk_size")) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(chunkSizeAttr))
        chunkSize = intAttr.getInt();
    }

    SmallVector<int64_t> targetMaskShape(*targetShape);
    VectorType maskTy = llvm::dyn_cast<VectorType>(mask.getType());
    VectorType offsetsTy = llvm::dyn_cast<VectorType>(offsets.getType());

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;
    SmallVector<Type> convertedOffsetTypes;
    SmallVector<Value> convertedOffsets;

    if (chunkSize > 1) {
      targetMaskShape.pop_back();
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = chunkSize / blockedChunkSize;
      chunkSize = blockedChunkSize;

      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedOffsetTypes = getUnrolledTypes(offsetsTy, targetMaskShape);

      SmallVector<Value> convertedMasksBase =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);
      SmallVector<Value> convertedOffsetsBase =
          pack(offsets, convertedOffsetTypes, targetMaskShape, loc, rewriter);

      for (auto maskVal : convertedMasksBase)
        convertedMasks.append(numNewChunks, maskVal);

      for (auto [baseOffset, offsetType] :
           llvm::zip(convertedOffsetsBase, convertedOffsetTypes)) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          Value inc = arith::ConstantIndexOp::create(rewriter, loc,
                                                     i * blockedChunkSize);
          Value incVec =
              vector::BroadcastOp::create(rewriter, loc, offsetType, inc);
          Value offsetVal =
              arith::AddIOp::create(rewriter, loc, baseOffset, incVec);
          convertedOffsets.push_back(offsetVal);
        }
      }
    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedMasks =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);

      convertedOffsetTypes = getUnrolledTypes(offsetsTy, *targetShape);
      convertedOffsets =
          pack(offsets, convertedOffsetTypes, *targetShape, loc, rewriter);
    }

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);

    auto layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    for (auto [v, o, m] :
         llvm::zip(convertedValues, convertedOffsets, convertedMasks)) {
      xegpu::StoreScatterOp::create(rewriter, loc, v, op.getDest(), o, m,
                                    rewriter.getI64IntegerAttr(chunkSize),
                                    op.getL1HintAttr(), op.getL2HintAttr(),
                                    op.getL3HintAttr(), layout);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollLoadMatrixOp : public UnrollPattern<xegpu::LoadMatrixOp> {
  using UnrollPattern<xegpu::LoadMatrixOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadMatrixOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    assert(valueTy && "the value type must be vector type!");

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != (size_t)valueTy.getRank())
      return failure();

    Type elemTy = valueTy.getElementType();
    ArrayRef<int64_t> shape = valueTy.getShape();
    auto layout = dyn_cast<xegpu::LayoutAttr>(op.getLayoutAttr());

    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(shape, *targetShape)) {
      auto adds = xegpu::addElementwise(
          rewriter, loc, mixedOffsets,
          getAsIndexOpFoldResult(op.getContext(), offsets));
      offsetsList.push_back(adds);
    }

    SmallVector<Value> newOps;
    layout = layout.dropInstData();
    for (SmallVector<OpFoldResult> offsets : offsetsList) {
      auto newOp = xegpu::LoadMatrixOp::create(
          rewriter, op.getLoc(), newValueTy, op.getMemDesc(), offsets, layout);
      newOps.push_back(newOp);
    }
    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreMatrixOp : public UnrollPattern<xegpu::StoreMatrixOp> {
  using UnrollPattern<xegpu::StoreMatrixOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreMatrixOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getData().getType());
    assert(valueTy && "the value type must be vector type!");
    ArrayRef<int64_t> shape = valueTy.getShape();
    auto layout = dyn_cast<xegpu::LayoutAttr>(op.getLayoutAttr());

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Value> convertedValues =
        pack(op.getData(), convertedValTypes, *targetShape, loc, rewriter);

    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(shape, *targetShape)) {
      auto adds = xegpu::addElementwise(
          rewriter, loc, mixedOffsets,
          getAsIndexOpFoldResult(op.getContext(), offsets));
      offsetsList.push_back(adds);
    }

    for (auto [v, offsets] : llvm::zip_equal(convertedValues, offsetsList))
      xegpu::StoreMatrixOp::create(rewriter, loc, v, op.getMemDesc(), offsets,
                                   layout.dropInstData());

    rewriter.eraseOp(op);
    return success();
  }
};

/// UnrollConvertLayoutOp pattern for unrolling xegpu::ConvertLayoutOp
/// operations. It first check whether the convert layout op has valid layouts
/// after inst_data stripped. If it does, it will unroll the vector into
/// multiple smaller vectors according to the target shape, and create multiple
/// ConvertLayoutOp with the unrolled vectors and the stripped layouts.
struct UnrollConvertLayoutOp : public UnrollPattern<xegpu::ConvertLayoutOp> {
  using UnrollPattern<xegpu::ConvertLayoutOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type valType = op.getType();

    xegpu::DistributeLayoutAttr inputLayout = op.getInputLayoutAttr();
    xegpu::DistributeLayoutAttr targetLayout = op.getTargetLayoutAttr();
    if (!inputLayout || !targetLayout)
      return rewriter.notifyMatchFailure(op, "missing layout attributes.");

    if (valType.isIntOrFloat()) {
      rewriter.replaceOp(op, op.getSource());
      assert(!inputLayout.dropInstData() && !targetLayout.dropInstData() &&
             "unexpected layout attributes for scalar type");
      return success();
    }

    if (inputLayout.getEffectiveInstDataAsInt().empty() ||
        targetLayout.getEffectiveInstDataAsInt().empty())
      return rewriter.notifyMatchFailure(op, "Not a target ConvertLayoutOp.");

    inputLayout = inputLayout.dropInstData();
    targetLayout = targetLayout.dropInstData();

    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    assert(valueTy && "the value type must be vector type!");

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != (size_t)valueTy.getRank())
      return failure();

    Value newSource = op.getSource();
    SmallVector<Value> newOps;
    if (inputLayout && targetLayout) {
      SmallVector<Type> convertedValTypes =
          getUnrolledTypes(valueTy, *targetShape);
      SmallVector<Value> convertedValues =
          pack(op.getOperand(), convertedValTypes, *targetShape, loc, rewriter);
      for (auto [v, t] : llvm::zip(convertedValues, convertedValTypes)) {
        auto newOp = xegpu::ConvertLayoutOp::create(rewriter, loc, t, v,
                                                    inputLayout, targetLayout);
        newOps.push_back(newOp);
      }
      newSource = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    }

    rewriter.replaceOp(op, newSource);
    return success();
  }
};

} // namespace

void mlir::xegpu::populateXeGPUUnrollPatterns(
    RewritePatternSet &patterns, const xegpu::UnrollOptions &options) {
  patterns.add<UnrollCreateNdOp, UnrollPrefetchNdOp, UnrollLoadNdOp,
               UnrollStoreNdOp, UnrollDpasOp, UnrollDpasMxOp,
               UnrollLoadMatrixOp, UnrollStoreMatrixOp, UnrollLoadGatherOp,
               UnrollStoreScatterOp, UnrollConvertLayoutOp>(
      patterns.getContext(), options);
}
