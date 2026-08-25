; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that llvm.powi on OpenCL targets maps to OpenCL.std pown. pown takes a
; 32-bit signed integer exponent, while llvm.powi is overloaded on the exponent
; type, so any other width must be converted with OpSConvert first -- emitting
; pown with e.g. a 64-bit operand produces a call no OpenCL consumer can
; resolve.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#F32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I16Ty:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64Ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#V4F32Ty:]] = OpTypeVector %[[#F32Ty]] 4

; An i32 exponent is already the right width: used directly, no conversion.
; CHECK-LABEL: Begin function test_powi_f32_i32
; CHECK: %[[#base32:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#exp32:]] = OpFunctionParameter %[[#I32Ty]]
; CHECK-NOT: OpSConvert
; CHECK: %[[#ret32:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] pown %[[#base32]] %[[#exp32]]
; CHECK: OpReturnValue %[[#ret32]]
define spir_func float @test_powi_f32_i32(float %x, i32 %n) {
  %res = call float @llvm.powi.f32.i32(float %x, i32 %n)
  ret float %res
}

; A wider exponent is narrowed to i32.
; CHECK-LABEL: Begin function test_powi_f32_i64
; CHECK: %[[#base64:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#exp64:]] = OpFunctionParameter %[[#I64Ty]]
; CHECK: %[[#conv64:]] = OpSConvert %[[#I32Ty]] %[[#exp64]]
; CHECK: %[[#ret64:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] pown %[[#base64]] %[[#conv64]]
; CHECK: OpReturnValue %[[#ret64]]
define spir_func float @test_powi_f32_i64(float %x, i64 %n) {
  %res = call float @llvm.powi.f32.i64(float %x, i64 %n)
  ret float %res
}

; A narrower exponent is widened to i32.
; CHECK-LABEL: Begin function test_powi_f32_i16
; CHECK: %[[#base16:]] = OpFunctionParameter %[[#F32Ty]]
; CHECK: %[[#exp16:]] = OpFunctionParameter %[[#I16Ty]]
; CHECK: %[[#conv16:]] = OpSConvert %[[#I32Ty]] %[[#exp16]]
; CHECK: %[[#ret16:]] = OpExtInst %[[#F32Ty]] %[[#ExtInstId]] pown %[[#base16]] %[[#conv16]]
; CHECK: OpReturnValue %[[#ret16]]
define spir_func float @test_powi_f32_i16(float %x, i16 %n) {
  %res = call float @llvm.powi.f32.i16(float %x, i16 %n)
  ret float %res
}

; The conversion is independent of the base being a vector.
; CHECK-LABEL: Begin function test_powi_v4f32_i64
; CHECK: %[[#baseV:]] = OpFunctionParameter %[[#V4F32Ty]]
; CHECK: %[[#expV:]] = OpFunctionParameter %[[#I64Ty]]
; CHECK: %[[#convV:]] = OpSConvert %[[#I32Ty]] %[[#expV]]
; CHECK: %[[#retV:]] = OpExtInst %[[#V4F32Ty]] %[[#ExtInstId]] pown %[[#baseV]] %[[#convV]]
; CHECK: OpReturnValue %[[#retV]]
define spir_func <4 x float> @test_powi_v4f32_i64(<4 x float> %x, i64 %n) {
  %res = call <4 x float> @llvm.powi.v4f32.i64(<4 x float> %x, i64 %n)
  ret <4 x float> %res
}
