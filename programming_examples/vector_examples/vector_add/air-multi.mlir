#map = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
module {
  func.func @vector_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c4) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
      %alloc = memref.alloc() : memref<256xf32, 2 : i32>
      %alloc_0 = memref.alloc() : memref<256xf32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<256xf32, 2 : i32>
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1024_2 = arith.constant 1024 : index
      scf.for %arg10 = %c0 to %c1024 step %c1024_2 {
        %0 = affine.apply #map()[%arg10, %arg4]
        %c256 = arith.constant 256 : index
        %c1_3 = arith.constant 1 : index
        air.dma_memcpy_nd (%alloc[] [] [], %arg7[%0] [%c256] [%c1_3]) : (memref<256xf32, 2 : i32>, memref<1024xf32>)
        %c256_4 = arith.constant 256 : index
        %c1_5 = arith.constant 1 : index
        air.dma_memcpy_nd (%alloc_0[] [] [], %arg8[%0] [%c256_4] [%c1_5]) : (memref<256xf32, 2 : i32>, memref<1024xf32>)
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c256_8 = arith.constant 256 : index
        linalg.add {op_name = "add_0"} ins(%alloc, %alloc_0 : memref<1024xf32, 2 : i32>, memref<1024xf32, 2 : i32>) outs(%alloc_1 : memref<1024xf32, 2 : i32>)
        %c256_9 = arith.constant 256 : index
        %c1_10 = arith.constant 1 : index
        air.dma_memcpy_nd (%arg9[%0] [%c256_9] [%c1_10], %alloc_1[] [] []) : (memref<1024xf32>, memref<256xf32, 2 : i32>)
        memref.dealloc %alloc : memref<256xf32, 2 : i32>
        memref.dealloc %alloc_0 : memref<256xf32, 2 : i32>
        memref.dealloc %alloc_1 : memref<256xf32, 2 : i32>
      }
    }
    return
  }
}

