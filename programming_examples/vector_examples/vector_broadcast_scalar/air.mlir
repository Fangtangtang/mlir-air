#map = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
module {
  func.func @vector_broadcast_scalar(%arg0: memref<65536xbf16>, %arg1: memref<65536x16xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1) : memref<65536xbf16>, memref<65536x16xbf16> {
      %alloc = memref.alloc() : memref<256x1xbf16, 2 : i32>
      %alloc_0 = memref.alloc() : memref<256x16xbf16, 2 : i32>
      %c0 = arith.constant 0 : index
      %c65536 = arith.constant 65536 : index
      %c512 = arith.constant 512 : index
      scf.for %arg8 = %c0 to %c65536 step %c512 {
        %0 = affine.apply #map()[%arg8, %arg3]
        %c256 = arith.constant 256 : index
        %c1_1 = arith.constant 1 : index
        air.dma_memcpy_nd (%alloc[] [] [], %arg6[%0] [%c256] [%c1_1]) : (memref<256x1xbf16, 2 : i32>, memref<65536xbf16>)
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c256_4 = arith.constant 256 : index
        scf.for %arg9 = %c0_2 to %c256_4 step %c1_3 {
          %subview = memref.subview %alloc[%arg9, %c0_2] [1, 1] [1, 1] : memref<256x1xbf16, 2 : i32> to memref<1x1xbf16, strided<[1, 1], offset: ?>, 2 : i32>
          %subview_10 = memref.subview %alloc_0[%arg9, %c0_2] [1, 16] [1, 1] : memref<256x16xbf16, 2 : i32> to memref<1x16xbf16, strided<[16, 1], offset: ?>, 2 : i32>
          %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<1x1xbf16, strided<[1, 1], offset: ?>, 2 : i32> into memref<1xbf16, strided<[1], offset: ?>, 2 : i32>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1]] : memref<1x16xbf16, strided<[16, 1], offset: ?>, 2 : i32> into memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
          %cst = arith.constant 0.000000e+00 : bf16
          %1 = memref.load %collapse_shape[%c0_2] : memref<1xbf16, strided<[1], offset: ?>, 2 : i32>
          %2 = vector.broadcast %1 : bf16 to vector<16xbf16>
          vector.transfer_write %2, %collapse_shape_11[%c0_2] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
        }
        %c0_5 = arith.constant 0 : index
        %c256_6 = arith.constant 256 : index
        %c16_7 = arith.constant 16 : index
        %c16_8 = arith.constant 16 : index
        %c1_9 = arith.constant 1 : index
        air.dma_memcpy_nd (%arg7[%0, %c0_5] [%c256_6, %c16_7] [%c16_8, %c1_9], %alloc_0[] [] []) : (memref<65536x16xbf16>, memref<256x16xbf16, 2 : i32>)
        memref.dealloc %alloc : memref<256x1xbf16, 2 : i32>
        memref.dealloc %alloc_0 : memref<256x16xbf16, 2 : i32>
      }
    }
    return
  }
}

