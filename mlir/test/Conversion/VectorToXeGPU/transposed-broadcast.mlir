// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

// vector.broadcast only prepends dims, so laying a lower-rank value out along
// the outer dims of the result takes a transpose. The conversion rewrites that
// into a shape_cast inserting the new dims where they belong, followed by a
// broadcast stretching them in place: no data movement, and no intermediate
// value whose XeGPU layout is the transpose of what its consumers want.

gpu.module @xevm_module {
gpu.func @row_vector_along_columns(%v: vector<16xf32>) -> vector<16x64xf32> {
  %b = vector.broadcast %v : vector<16xf32> to vector<64x16xf32>
  %t = vector.transpose %b, [1, 0] : vector<64x16xf32> to vector<16x64xf32>
  gpu.return %t : vector<16x64xf32>
}

// CHECK-LABEL: @row_vector_along_columns(
// CHECK-SAME:    %[[V:.+]]: vector<16xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[V]] : vector<16xf32> to vector<16x1xf32>
// CHECK:         %[[RES:.+]] = vector.broadcast %[[CAST]] : vector<16x1xf32> to vector<16x64xf32>
// CHECK-NOT:     vector.transpose
// CHECK:         gpu.return %[[RES]] : vector<16x64xf32>
}

// -----

// Two prepended dims end up as two trailing unit dims.
gpu.module @xevm_module {
gpu.func @two_broadcast_dims(%v: vector<8xf32>) -> vector<8x4x2xf32> {
  %b = vector.broadcast %v : vector<8xf32> to vector<4x2x8xf32>
  %t = vector.transpose %b, [2, 0, 1] : vector<4x2x8xf32> to vector<8x4x2xf32>
  gpu.return %t : vector<8x4x2xf32>
}

// CHECK-LABEL: @two_broadcast_dims(
// CHECK-SAME:    %[[V:.+]]: vector<8xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[V]] : vector<8xf32> to vector<8x1x1xf32>
// CHECK:         %[[RES:.+]] = vector.broadcast %[[CAST]] : vector<8x1x1xf32> to vector<8x4x2xf32>
// CHECK-NOT:     vector.transpose
// CHECK:         gpu.return %[[RES]] : vector<8x4x2xf32>
}

// -----

// A rank-2 source keeps its own dim order; only the prepended dim moves.
gpu.module @xevm_module {
gpu.func @rank2_source(%v: vector<8x16xf32>) -> vector<8x16x4xf32> {
  %b = vector.broadcast %v : vector<8x16xf32> to vector<4x8x16xf32>
  %t = vector.transpose %b, [1, 2, 0] : vector<4x8x16xf32> to vector<8x16x4xf32>
  gpu.return %t : vector<8x16x4xf32>
}

// CHECK-LABEL: @rank2_source(
// CHECK-SAME:    %[[V:.+]]: vector<8x16xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<8x16x1xf32>
// CHECK:         %[[RES:.+]] = vector.broadcast %[[CAST]] : vector<8x16x1xf32> to vector<8x16x4xf32>
// CHECK-NOT:     vector.transpose
// CHECK:         gpu.return %[[RES]] : vector<8x16x4xf32>
}

// -----

// The source's own dims are shuffled, so the transpose is real data movement
// and has to stay.
gpu.module @xevm_module {
gpu.func @source_dims_shuffled(%v: vector<8x16xf32>) -> vector<16x8x4xf32> {
  %b = vector.broadcast %v : vector<8x16xf32> to vector<4x8x16xf32>
  %t = vector.transpose %b, [2, 1, 0] : vector<4x8x16xf32> to vector<16x8x4xf32>
  gpu.return %t : vector<16x8x4xf32>
}

// CHECK-LABEL: @source_dims_shuffled(
// CHECK:         %[[B:.+]] = vector.broadcast %{{.+}} : vector<8x16xf32> to vector<4x8x16xf32>
// CHECK:         %[[RES:.+]] = vector.transpose %[[B]], [2, 1, 0]
// CHECK:         gpu.return %[[RES]] : vector<16x8x4xf32>
}

// -----

// The broadcast dim stays outermost, so there is nothing to rewrite: a
// shape_cast cannot put a unit dim in front and have the broadcast stretch it
// into the result's leading position without moving the source's data.
gpu.module @xevm_module {
gpu.func @broadcast_dim_stays_outer(%v: vector<8x16xf32>) -> vector<4x16x8xf32> {
  %b = vector.broadcast %v : vector<8x16xf32> to vector<4x8x16xf32>
  %t = vector.transpose %b, [0, 2, 1] : vector<4x8x16xf32> to vector<4x16x8xf32>
  gpu.return %t : vector<4x16x8xf32>
}

// CHECK-LABEL: @broadcast_dim_stays_outer(
// CHECK:         %[[B:.+]] = vector.broadcast %{{.+}} : vector<8x16xf32> to vector<4x8x16xf32>
// CHECK:         %[[RES:.+]] = vector.transpose %[[B]], [0, 2, 1]
// CHECK:         gpu.return %[[RES]] : vector<4x16x8xf32>
}

// -----

// The broadcast stretched a source dim in place, so that dim is no longer a
// unit dim a shape_cast could re-create, and the transpose has to stay.
gpu.module @xevm_module {
gpu.func @stretched_source_dim(%v: vector<1x16xf32>) -> vector<8x16x4xf32> {
  %b = vector.broadcast %v : vector<1x16xf32> to vector<4x8x16xf32>
  %t = vector.transpose %b, [1, 2, 0] : vector<4x8x16xf32> to vector<8x16x4xf32>
  gpu.return %t : vector<8x16x4xf32>
}

// CHECK-LABEL: @stretched_source_dim(
// CHECK:         %[[B:.+]] = vector.broadcast %{{.+}} : vector<1x16xf32> to vector<4x8x16xf32>
// CHECK:         %[[RES:.+]] = vector.transpose %[[B]], [1, 2, 0]
// CHECK:         gpu.return %[[RES]] : vector<8x16x4xf32>
}
