#map = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
module {
  func.func @eltwise_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
      air.segment @segment_0  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
        %alloc = memref.alloc() : memref<1024xf32, 1 : i32>
        %alloc_1 = memref.alloc() : memref<1024xf32, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1024xf32, 1 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg10[] [] []) : (memref<1024xf32, 1 : i32>, memref<1024xf32>)
        air.dma_memcpy_nd (%alloc_1[] [] [], %arg11[] [] []) : (memref<1024xf32, 1 : i32>, memref<1024xf32>)
        %c1_3 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        air.herd @herd_0  tile (%arg13, %arg14) in (%arg15=%c1_3, %arg16=%c4) args(%arg17=%alloc, %arg18=%alloc_1, %arg19=%alloc_2) : memref<1024xf32, 1 : i32>, memref<1024xf32, 1 : i32>, memref<1024xf32, 1 : i32> {
          %alloc_4 = memref.alloc() : memref<256xf32, 2 : i32>
          %alloc_5 = memref.alloc() : memref<256xf32, 2 : i32>
          %alloc_6 = memref.alloc() : memref<256xf32, 2 : i32>
          %c0 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1024_7 = arith.constant 1024 : index
          scf.for %arg20 = %c0 to %c1024 step %c1024_7 {
            %0 = affine.apply #map()[%arg20, %arg14]
            %c256 = arith.constant 256 : index
            %c1_8 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc_4[] [] [], %arg17[%0] [%c256] [%c1_8]) : (memref<256xf32, 2 : i32>, memref<1024xf32, 1 : i32>)
            %c256_9 = arith.constant 256 : index
            %c1_10 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc_5[] [] [], %arg18[%0] [%c256_9] [%c1_10]) : (memref<256xf32, 2 : i32>, memref<1024xf32, 1 : i32>)
            %c0_11 = arith.constant 0 : index
            %c256_12 = arith.constant 256 : index
            %c1_13 = arith.constant 1 : index
            scf.for %arg21 = %c0_11 to %c256_12 step %c1_13 {
              %1 = memref.load %alloc_4[%arg21] : memref<256xf32, 2 : i32>
              %2 = memref.load %alloc_5[%arg21] : memref<256xf32, 2 : i32>
              %3 = arith.addf %1, %2 : f32
              memref.store %3, %alloc_6[%arg21] : memref<256xf32, 2 : i32>
            }
            %c256_14 = arith.constant 256 : index
            %c1_15 = arith.constant 1 : index
            air.dma_memcpy_nd (%arg19[%0] [%c256_14] [%c1_15], %alloc_6[] [] []) : (memref<1024xf32, 1 : i32>, memref<256xf32, 2 : i32>)
            memref.dealloc %alloc_4 : memref<256xf32, 2 : i32>
            memref.dealloc %alloc_5 : memref<256xf32, 2 : i32>
            memref.dealloc %alloc_6 : memref<256xf32, 2 : i32>
          }
        }
        air.dma_memcpy_nd (%arg12[] [] [], %alloc_2[] [] []) : (memref<1024xf32>, memref<1024xf32, 1 : i32>)
        memref.dealloc %alloc : memref<1024xf32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1024xf32, 1 : i32>
        memref.dealloc %alloc_2 : memref<1024xf32, 1 : i32>
      }
    }
    return
  }
}

