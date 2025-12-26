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
        %alloc = memref.alloc() : memref<4x1x64x256xi8, 1 : i32>
        %alloc_0 = memref.alloc() : memref<1x4x256x128xi8, 1 : i32>
        %alloc_1 = memref.alloc() : memref<4x4x64x128xi16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x1x8x8x8x8xi8, 2 : i32>
        %alloc_3 = memref.alloc() : memref<1x1x16x8x8x8xi8, 2 : i32>
        %alloc_4 = memref.alloc() : memref<4x4x16x8x8x8xi16, 2 : i32>
        %0 = affine.apply #map()[%arg10]
        %1 = affine.apply #map1()[%arg11]
        %c4 = arith.constant 4 : index
        %c4_5 = arith.constant 4 : index
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4_5) args(%arg19=%alloc_2, %arg20=%alloc_3, %arg21=%alloc_4, %arg22=%alloc, %arg23=%alloc_0) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32> {
          %subview = memref.subview %arg21[%arg15, %arg16, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<4x4x16x8x8x8xi16, 2 : i32> to memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
          %c0_i16 = arith.constant 0 : i16
          linalg.fill ins(%c0_i16 : i16) outs(%subview : memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>)
        }
        %c0 = arith.constant 0 : index
        %c2_6 = arith.constant 2 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg15 = %c0 to %c2_6 step %c1_7 {
          %2 = affine.apply #map()[%arg15]
          %c0_20 = arith.constant 0 : index
          %c0_21 = arith.constant 0 : index
          %c4_22 = arith.constant 4 : index
          %c1_23 = arith.constant 1 : index
          %c64_24 = arith.constant 64 : index
          %c256_25 = arith.constant 256 : index
          %c32768_26 = arith.constant 32768 : index
          %c256_27 = arith.constant 256 : index
          %c512_28 = arith.constant 512 : index
          %c1_29 = arith.constant 1 : index
          air.dma_memcpy_nd (%alloc[] [] [], %arg12[%c0_20, %c0_21, %0, %2] [%c4_22, %c1_23, %c64_24, %c256_25] [%c32768_26, %c256_27, %c512_28, %c1_29]) : (memref<4x1x64x256xi8, 1 : i32>, memref<512x512xi8>)
          %c0_30 = arith.constant 0 : index
          %c0_31 = arith.constant 0 : index
          %c1_32 = arith.constant 1 : index
          %c4_33 = arith.constant 4 : index
          %c256_34 = arith.constant 256 : index
          %c128_35 = arith.constant 128 : index
          %c131072 = arith.constant 131072 : index
          %c128_36 = arith.constant 128 : index
          %c512_37 = arith.constant 512 : index
          %c1_38 = arith.constant 1 : index
          air.dma_memcpy_nd (%alloc_0[] [] [], %arg13[%c0_30, %c0_31, %2, %1] [%c1_32, %c4_33, %c256_34, %c128_35] [%c131072, %c128_36, %c512_37, %c1_38]) : (memref<1x4x256x128xi8, 1 : i32>, memref<512x512xi8>)
          %c4_39 = arith.constant 4 : index
          %c4_40 = arith.constant 4 : index
          air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_39, %arg19=%c4_40) args(%arg20=%alloc_2, %arg21=%alloc_3, %arg22=%alloc_4, %arg23=%alloc, %arg24=%alloc_0) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32> {
            %c0_41 = arith.constant 0 : index
            %c4_42 = arith.constant 4 : index
            %c1_43 = arith.constant 1 : index
            scf.for %arg25 = %c0_41 to %c4_42 step %c1_43 {
              %3 = affine.apply #map2()[%arg25]
              %c0_44 = arith.constant 0 : index
              %c0_45 = arith.constant 0 : index
              %c0_46 = arith.constant 0 : index
              %c0_47 = arith.constant 0 : index
              %c1_48 = arith.constant 1 : index
              %c1_49 = arith.constant 1 : index
              %c8 = arith.constant 8 : index
              %c8_50 = arith.constant 8 : index
              %c8_51 = arith.constant 8 : index
              %c8_52 = arith.constant 8 : index
              %c16384 = arith.constant 16384 : index
              %c16384_53 = arith.constant 16384 : index
              %c8_54 = arith.constant 8 : index
              %c2048 = arith.constant 2048 : index
              %c256_55 = arith.constant 256 : index
              %c1_56 = arith.constant 1 : index
              air.dma_memcpy_nd (%arg20[] [] [], %arg23[%arg16, %c0_44, %c0_45, %c0_46, %c0_47, %3] [%c1_48, %c1_49, %c8, %c8_50, %c8_51, %c8_52] [%c16384, %c16384_53, %c8_54, %c2048, %c256_55, %c1_56]) : (memref<1x1x8x8x8x8xi8, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>)
              %c0_57 = arith.constant 0 : index
              %c0_58 = arith.constant 0 : index
              %c0_59 = arith.constant 0 : index
              %c0_60 = arith.constant 0 : index
              %c1_61 = arith.constant 1 : index
              %c1_62 = arith.constant 1 : index
              %c16 = arith.constant 16 : index
              %c8_63 = arith.constant 8 : index
              %c8_64 = arith.constant 8 : index
              %c8_65 = arith.constant 8 : index
              %c131072_66 = arith.constant 131072 : index
              %c32768_67 = arith.constant 32768 : index
              %c8_68 = arith.constant 8 : index
              %c1024 = arith.constant 1024 : index
              %c128_69 = arith.constant 128 : index
              %c1_70 = arith.constant 1 : index
              air.dma_memcpy_nd (%arg21[] [] [], %arg24[%c0_57, %arg17, %c0_58, %c0_59, %3, %c0_60] [%c1_61, %c1_62, %c16, %c8_63, %c8_64, %c8_65] [%c131072_66, %c32768_67, %c8_68, %c1024, %c128_69, %c1_70]) : (memref<1x1x16x8x8x8xi8, 2 : i32>, memref<1x4x256x128xi8, 1 : i32>)
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
        %c4_8 = arith.constant 4 : index
        %c4_9 = arith.constant 4 : index
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c4_8, %arg18=%c4_9) args(%arg19=%alloc_2, %arg20=%alloc_3, %arg21=%alloc_4, %arg22=%alloc, %arg23=%alloc_0, %arg24=%alloc_1) : memref<1x1x8x8x8x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32>, memref<4x4x64x128xi16, 1 : i32> {
          %subview = memref.subview %arg21[%arg15, %arg16, 0, 0, 0, 0] [1, 1, 16, 8, 8, 8] [1, 1, 1, 1, 1, 1] : memref<4x4x16x8x8x8xi16, 2 : i32> to memref<1x1x16x8x8x8xi16, strided<[32768, 8192, 512, 64, 8, 1], offset: ?>, 2 : i32>
          %c0_20 = arith.constant 0 : index
          %c0_21 = arith.constant 0 : index
          %c1_22 = arith.constant 1 : index
          %c1_23 = arith.constant 1 : index
          %c64_24 = arith.constant 64 : index
          %c128_25 = arith.constant 128 : index
          %c32768_26 = arith.constant 32768 : index
          %c8192_27 = arith.constant 8192 : index
          %c128_28 = arith.constant 128 : index
          %c1_29 = arith.constant 1 : index
          %c0_30 = arith.constant 0 : index
          %c0_31 = arith.constant 0 : index
          %c0_32 = arith.constant 0 : index
          %c0_33 = arith.constant 0 : index
          %c1_34 = arith.constant 1 : index
          %c1_35 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c8_36 = arith.constant 8 : index
          %c16 = arith.constant 16 : index
          %c8_37 = arith.constant 8 : index
          %c32768_38 = arith.constant 32768 : index
          %c8192_39 = arith.constant 8192 : index
          %c64_40 = arith.constant 64 : index
          %c8_41 = arith.constant 8 : index
          %c512_42 = arith.constant 512 : index
          %c1_43 = arith.constant 1 : index
          air.dma_memcpy_nd (%arg24[%arg15, %arg16, %c0_20, %c0_21] [%c1_22, %c1_23, %c64_24, %c128_25] [%c32768_26, %c8192_27, %c128_28, %c1_29], %arg21[%arg15, %arg16, %c0_30, %c0_31, %c0_32, %c0_33] [%c1_34, %c1_35, %c8, %c8_36, %c16, %c8_37] [%c32768_38, %c8192_39, %c64_40, %c8_41, %c512_42, %c1_43]) : (memref<4x4x64x128xi16, 1 : i32>, memref<4x4x16x8x8x8xi16, 2 : i32>)
        }
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c512_10 = arith.constant 512 : index
        %c1_11 = arith.constant 1 : index
        %c0_12 = arith.constant 0 : index
        %c0_13 = arith.constant 0 : index
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %c4_16 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c4_17 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c32768 = arith.constant 32768 : index
        %c128_18 = arith.constant 128 : index
        %c8192 = arith.constant 8192 : index
        %c1_19 = arith.constant 1 : index
        air.dma_memcpy_nd (%arg14[%0, %1] [%c256, %c512] [%c512_10, %c1_11], %alloc_1[%c0_12, %c0_13, %c0_14, %c0_15] [%c4_16, %c64, %c4_17, %c128] [%c32768, %c128_18, %c8192, %c1_19]) : (memref<512x512xi16>, memref<4x4x64x128xi16, 1 : i32>)
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

