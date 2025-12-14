#map = affine_map<()[s0, s1] -> (s0 + s1 * 1024)>
module {
  func.func @vector_add(%arg0: memref<65536xf32>, %arg1: memref<65536xf32>, %arg2: memref<65536xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<65536xf32>, memref<65536xf32>, memref<65536xf32> {
      %alloc = memref.alloc() : memref<1024xf32, 2 : i32>
      %alloc_0 = memref.alloc() : memref<1024xf32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<1024xf32, 2 : i32>
      %c0 = arith.constant 0 : index
      %c65536 = arith.constant 65536 : index
      %c2048 = arith.constant 2048 : index
      scf.for %arg10 = %c0 to %c65536 step %c2048 {
        %0 = affine.apply #map()[%arg10, %arg4]
        %c1024 = arith.constant 1024 : index
        %c1_2 = arith.constant 1 : index
        air.dma_memcpy_nd (%alloc[] [] [], %arg7[%0] [%c1024] [%c1_2]) : (memref<1024xf32, 2 : i32>, memref<65536xf32>)
        %c1024_3 = arith.constant 1024 : index
        %c1_4 = arith.constant 1 : index
        air.dma_memcpy_nd (%alloc_0[] [] [], %arg8[%0] [%c1024_3] [%c1_4]) : (memref<1024xf32, 2 : i32>, memref<65536xf32>)
        %c0_5 = arith.constant 0 : index
        %c1_6 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c1024_7 = arith.constant 1024 : index
        scf.for %arg11 = %c0_5 to %c1024_7 step %c1_6 {
          %1 = memref.load %alloc[%arg11] : memref<1024xf32, 2 : i32>
          %2 = memref.load %alloc_0[%arg11] : memref<1024xf32, 2 : i32>
          %3 = arith.addf %1, %2 : f32
          memref.store %3, %alloc_1[%arg11] : memref<1024xf32, 2 : i32>
        }
        %c1024_8 = arith.constant 1024 : index
        %c1_9 = arith.constant 1 : index
        air.dma_memcpy_nd (%arg9[%0] [%c1024_8] [%c1_9], %alloc_1[] [] []) : (memref<65536xf32>, memref<1024xf32, 2 : i32>)
        memref.dealloc %alloc : memref<1024xf32, 2 : i32>
        memref.dealloc %alloc_0 : memref<1024xf32, 2 : i32>
        memref.dealloc %alloc_1 : memref<1024xf32, 2 : i32>
      }
    }
    return
  }
}
