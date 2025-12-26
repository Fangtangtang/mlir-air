#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>
module {
  func.func @matmul_bf16(%arg0: memref<512x512xi8>, %arg1: memref<512x512xi8>, %arg2: memref<512x512xi16>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c2, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<512x512xi8>, memref<512x512xi8>, memref<512x512xi16> {
      air.segment @matmul_seg  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<512x512xi8>, memref<512x512xi8>, memref<512x512xi16> {
        %alloc = memref.alloc() : memref<4x1x64x256xi8, 1 : i32> // L2 A
        %alloc_0 = memref.alloc() : memref<1x4x256x128xi8, 1 : i32>
        %alloc_1 = memref.alloc() : memref<4x4x64x128xi16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x1x8x8x8x8xi8, 2 : i32>
        %alloc_3 = memref.alloc() : memref<1x1x16x8x8x8xi8, 2 : i32>
        %alloc_4 = memref.alloc() : memref<4x4x16x8x8x8xi16, 2 : i32> // L1 C (distributed to 4x4)
        %0 = affine.apply #map()[%arg10]
        %1 = affine.apply #map1()[%arg11]
        // init zero
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=4, %arg18=4) args(%arg19=%alloc_2, %arg20=%alloc_3, %arg21=%alloc_4, %arg22=%alloc, %arg23=%alloc_0) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32> {
          %subview = memref.subview %arg21[%arg15, %arg16, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<4x4x16x8x8x8xi16, 2 : i32> to memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
          %c0_i16 = arith.constant 0 : i16
          linalg.fill ins(%c0_i16 : i16) outs(%subview : memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>)
        }
        scf.for %arg15 = 0 to 2 step 1 {
          %2 = affine.apply #map()[%arg15]
          air.dma_memcpy_nd (%alloc[] [] [], %arg12[0, 0, %0, %2] [4, 1, 64 256] [32768, 256, 512, 1]) : (memref<4x1x64x256xi8, 1 : i32>, memref<512x512xi8>)
          air.dma_memcpy_nd (%alloc_0[] [] [], %arg13[0, 0, %2, %1] [1, 4, 256, 128] [131072, 128, 512, 1]) : (memref<1x4x256x128xi8, 1 : i32>, memref<512x512xi8>)
          air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=4, %arg19=4) args(%arg20=%alloc_2, %arg21=%alloc_3, %arg22=%alloc_4, %arg23=%alloc, %arg24=%alloc_0) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32> {
            // 4 iterations (64*4)
            scf.for %arg25 = 0 to 4 step 1 {
              %3 = affine.apply #map2()[%arg25]
              // copy A tile from L2 to L1 (64x64 in each compute tile)
              air.dma_memcpy_nd (%arg20[] [] [], %arg23[%arg16, 0, 0, 0, 0, %3] [1, 1, 8, 8, 8, 8] [16384, 16384, 8, 2048, 256, 1]) : (memref<1x1x8x8x8x8xi8, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>)
              // copy B tile from L2 to L1 (64x128 in each compute tile)
              air.dma_memcpy_nd (%arg21[] [] [], %arg24[0, %arg17, 0, 0, %3, 0] [1, 1, 16, 8, 8, 8] [131072, 32768, 8, 1024, 128, 1]) : (memref<1x1x16x8x8x8xi8, 2 : i32>, memref<1x4x256x128xi8, 1 : i32>)
              %subview = memref.subview %arg22[%arg16, %arg17, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<4x4x16x8x8x8xi16, 2 : i32> to memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
              linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg20, %arg21 : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>) outs(%subview : memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>) {
              ^bb0(%in: i8, %in_71: i8, %out: i16):
                %4 = arith.extsi %in : i8 to i16
                %5 = arith.extsi %in_71 : i8 to i16
                %6 = arith.muli %4, %5 : i16
                %7 = arith.addi %out, %6 : i16
                linalg.yield %7 : i16
              }
            }
          }
        }
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=4, %arg18=4) args(%arg19=%alloc_2, %arg20=%alloc_3, %arg21=%alloc_4, %arg22=%alloc, %arg23=%alloc_0, %arg24=%alloc_1) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32>, memref<4x4x64x128xi16, 1 : i32> {
          %subview = memref.subview %arg21[%arg15, %arg16, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<4x4x16x8x8x8xi16, 2 : i32> to memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
          air.dma_memcpy_nd (%arg24[%arg15, %arg16, 0 0] [1, 1, 64, 128] [32768, 8192, 128, 1], %arg21[%arg15, %arg16, 0, 0, 0, 0] [1, 1, 8, 8, 16, 8] [32768, 8192, 64, 8, 512, 1]) : (memref<4x4x64x128xi16, 1 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>)
        }
        air.dma_memcpy_nd (%arg14[%0, %1] [256, 512] [512, 1], %alloc_1[0, 0, 0, 0] [4, 64, 4, 128] [32768, 128, 8192, 1]) : (memref<512x512xi16>, memref<4x4x64x128xi16, 1 : i32>)
        memref.dealloc %alloc : memref<4x1x64x256xi8, 1 : i32>
        memref.dealloc %alloc_0 : memref<1x4x256x128xi8, 1 : i32>
        memref.dealloc %alloc_1 : memref<4x4x64x128xi16, 1 : i32>
        memref.dealloc %alloc_2 : memref<1x1x8x8x8x8xi8, 2 : i32>
        memref.dealloc %alloc_3 : memref<1x1x16x8x8x8xi8, 2 : i32>
        memref.dealloc %alloc_4 : memref<4x4x16x8x8x8xi16, 2 : i32>
      }
    }
    return
  }
}

