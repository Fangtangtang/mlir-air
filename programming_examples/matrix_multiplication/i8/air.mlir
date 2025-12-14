python3 /home/sf668/workspace/mlir-air/programming_examples/matrix_multiplication/i8/run.py -p --arch aie2 --direct-codegen
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 + 1)>
#map4 = affine_map<(d0) -> (d0 * 32)>
#map5 = affine_map<(d0) -> (d0 * 512)>
#map6 = affine_map<()[s0] -> (s0 * 512 + 512)>
#map7 = affine_map<()[s0] -> (s0 * 32 + 32)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @matmul_bf16(%arg0: memref<512x512xi8>, %arg1: memref<512x512xi8>, %arg2: memref<512x512xi16>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c2, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<512x512xi8>, memref<512x512xi8>, memref<512x512xi16> {
      air.segment @matmul_seg  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<512x512xi8>, memref<512x512xi8>, memref<512x512xi16> {
        %c8192 = arith.constant 8192 : index
        %c131072 = arith.constant 131072 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c32768 = arith.constant 32768 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        %c1_0 = arith.constant 1 : index
        %c2_1 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<4x1x64x256xi8, 1 : i32>
        %alloc_2 = memref.alloc() : memref<1x4x256x128xi8, 1 : i32>
        %alloc_3 = memref.alloc() : memref<4x4x64x128xi16, 1 : i32>
        %alloc_4 = memref.alloc() : memref<1x1x8x16x4x8xi8, 2 : i32>
        %alloc_5 = memref.alloc() : memref<1x1x16x8x8x8xi8, 2 : i32>
        %alloc_6 = memref.alloc() : memref<4x4x16x16x4x8xi16, 2 : i32>
        %0 = affine.apply #map()[%arg10]
        %1 = affine.apply #map1()[%arg11]
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4) args(%arg19=%alloc_6) : memref<4x4x16x16x4x8xi16, 2 : i32> {
          %cst = arith.constant dense<0> : vector<1x1x1x1x4x8xi16>
          %c1_7 = arith.constant 1 : index
          %c16 = arith.constant 16 : index
          %c0_8 = arith.constant 0 : index
          scf.for %arg20 = %c0_8 to %c16 step %c1_7 {
            scf.for %arg21 = %c0_8 to %c16 step %c1_7 {
              vector.transfer_write %cst, %arg19[%arg15, %arg16, %arg20, %arg21, %c0_8, %c0_8] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x8xi16>, memref<4x4x16x16x4x8xi16, 2 : i32>
            }
          }
        }
        scf.for %arg15 = %c0 to %c2_1 step %c1_0 {
          %2 = affine.apply #map()[%arg15]
          air.dma_memcpy_nd (%alloc[] [] [], %arg12[%c0, %c0, %0, %2] [%c4, %c1_0, %c64, %c256] [%c32768, %c256, %c512, %c1_0]) : (memref<4x1x64x256xi8, 1 : i32>, memref<512x512xi8>)
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg13[%c0, %c0, %2, %1] [%c1_0, %c4, %c256, %c128] [%c131072, %c128, %c512, %c1_0]) : (memref<1x4x256x128xi8, 1 : i32>, memref<512x512xi8>)
          air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%arg20=%alloc_4, %arg21=%alloc_5, %arg22=%alloc_6, %arg23=%alloc, %arg24=%alloc_2) : memref<1x1x8x16x4x8xi8, 2 : i32>, memref<1x1x16x8x8x8xi8, 2 : i32>, memref<4x4x16x16x4x8xi16, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>, memref<1x4x256x128xi8, 1 : i32> {
            %c64_7 = arith.constant 64 : index
            %c512_8 = arith.constant 512 : index
            %3 = ub.poison : i16
            %4 = ub.poison : i8
            %c2_9 = arith.constant 2 : index
            %c128_10 = arith.constant 128 : index
            %c32768_11 = arith.constant 32768 : index
            %c131072_12 = arith.constant 131072 : index
            %c256_13 = arith.constant 256 : index
            %c1024 = arith.constant 1024 : index
            %c16384 = arith.constant 16384 : index
            %c16 = arith.constant 16 : index
            %c8 = arith.constant 8 : index
            %c0_14 = arith.constant 0 : index
            %c4_15 = arith.constant 4 : index
            %c1_16 = arith.constant 1 : index
            scf.for %arg25 = %c0_14 to %c4_15 step %c1_16 {
              %5 = affine.apply #map2()[%arg25]
              air.dma_memcpy_nd (%arg20[] [] [], %arg23[%arg16, %c0_14, %c0_14, %c0_14, %c0_14, %5] [%c1_16, %c1_16, %c8, %c16, %c4_15, %c8] [%c16384, %c16384, %c8, %c1024, %c256_13, %c1_16]) : (memref<1x1x8x16x4x8xi8, 2 : i32>, memref<4x1x64x256xi8, 1 : i32>)
              air.dma_memcpy_nd (%arg21[] [] [], %arg24[%c0_14, %arg17, %c0_14, %c0_14, %5, %c0_14] [%c1_16, %c1_16, %c16, %c8, %c8, %c8] [%c131072_12, %c32768_11, %c8, %c1024, %c128_10, %c1_16]) : (memref<1x1x16x8x8x8xi8, 2 : i32>, memref<1x4x256x128xi8, 1 : i32>)
              scf.for %arg26 = %c0_14 to %c16 step %c2_9 {
                scf.for %arg27 = %c0_14 to %c16 step %c2_9 {
                  %6 = vector.transfer_read %arg22[%arg16, %arg17, %arg27, %arg26, %c0_14, %c0_14], %3 {in_bounds = [true, true, true, true]} : memref<4x4x16x16x4x8xi16, 2 : i32>, vector<1x1x4x8xi16>
                  %7 = affine.apply #map3()[%arg27]
                  %8 = vector.transfer_read %arg22[%arg16, %arg17, %7, %arg26, %c0_14, %c0_14], %3 {in_bounds = [true, true, true, true]} : memref<4x4x16x16x4x8xi16, 2 : i32>, vector<1x1x4x8xi16>
                  %9 = affine.apply #map3()[%arg26]
                  %10 = vector.transfer_read %arg22[%arg16, %arg17, %arg27, %9, %c0_14, %c0_14], %3 {in_bounds = [true, true, true, true]} : memref<4x4x16x16x4x8xi16, 2 : i32>, vector<1x1x4x8xi16>
                  %11 = affine.apply #map3()[%arg27]
                  %12 = affine.apply #map3()[%arg26]
                  %13 = vector.transfer_read %arg22[%arg16, %arg17, %11, %12, %c0_14, %c0_14], %3 {in_bounds = [true, true, true, true]} : memref<4x4x16x16x4x8xi16, 2 : i32>, vector<1x1x4x8xi16>
                  %14 = vector.shape_cast %6 : vector<1x1x4x8xi16> to vector<32xi16>
                  %15 = vector.shape_cast %8 : vector<1x1x4x8xi16> to vector<32xi16>
                  %16 = vector.shape_cast %10 : vector<1x1x4x8xi16> to vector<32xi16>
                  %17 = vector.shape_cast %13 : vector<1x1x4x8xi16> to vector<32xi16>
                  %collapse_shape = memref.collapse_shape %arg20 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x16x4x8xi8, 2 : i32> into memref<4096xi8, 2 : i32>
                  %18 = affine.apply #map4(%arg26)
                  %collapse_shape_17 = memref.collapse_shape %arg21 [[0, 1, 2, 3, 4, 5]] : memref<1x1x16x8x8x8xi8, 2 : i32> into memref<8192xi8, 2 : i32>
                  %19 = affine.apply #map5(%arg27)
                  %collapse_shape_18 = memref.collapse_shape %arg21 [[0, 1, 2, 3, 4, 5]] : memref<1x1x16x8x8x8xi8, 2 : i32> into memref<8192xi8, 2 : i32>
                  %20 = affine.apply #map6()[%arg27]
                  %collapse_shape_19 = memref.collapse_shape %arg20 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x16x4x8xi8, 2 : i32> into memref<4096xi8, 2 : i32>
                  %21 = affine.apply #map7()[%arg26]
                  %22 = arith.extsi %14 : vector<32xi16> to vector<32xi32>
                  %23 = arith.extsi %15 : vector<32xi16> to vector<32xi32>
                  %24 = arith.extsi %16 : vector<32xi16> to vector<32xi32>
                  %25 = arith.extsi %17 : vector<32xi16> to vector<32xi32>
                  %26:8 = scf.for %arg28 = %c0_14 to %c8 step %c1_16 iter_args(%arg29 = %22, %arg30 = %23, %arg31 = %24, %arg32 = %25, %arg33 = %18, %arg34 = %19, %arg35 = %20, %arg36 = %21) -> (vector<32xi32>, vector<32xi32>, vector<32xi32>, vector<32xi32>, index, index, index, index) {
                    %39 = vector.shape_cast %arg29 : vector<32xi32> to vector<1x1x4x8xi32>
                    %40 = vector.shape_cast %arg30 : vector<32xi32> to vector<1x1x4x8xi32>
                    %41 = vector.shape_cast %arg31 : vector<32xi32> to vector<1x1x4x8xi32>
                    %42 = vector.shape_cast %arg32 : vector<32xi32> to vector<1x1x4x8xi32>
                    %43 = vector.transfer_read %collapse_shape[%arg33], %4 {in_bounds = [true]} : memref<4096xi8, 2 : i32>, vector<32xi8>
                    %44 = vector.shape_cast %43 : vector<32xi8> to vector<1x1x4x8xi8>
                    %45 = arith.addi %arg33, %c512_8 : index
                    %46 = vector.transfer_read %collapse_shape_17[%arg34], %4 {in_bounds = [true]} : memref<8192xi8, 2 : i32>, vector<64xi8>
                    %47 = vector.shape_cast %46 : vector<64xi8> to vector<1x1x8x8xi8>
                    %48 = arith.addi %arg34, %c64_7 : index
                    %49 = arith.extsi %44 : vector<1x1x4x8xi8> to vector<1x1x4x8xi16>
                    %50 = arith.extsi %47 : vector<1x1x8x8xi8> to vector<1x1x8x8xi16>
                    %51 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %49, %50, %39 : vector<1x1x4x8xi16>, vector<1x1x8x8xi16> into vector<1x1x4x8xi32>
                    %52 = vector.transfer_read %collapse_shape_18[%arg35], %4 {in_bounds = [true]} : memref<8192xi8, 2 : i32>, vector<64xi8>
                    %53 = vector.shape_cast %52 : vector<64xi8> to vector<1x1x8x8xi8>
                    %54 = arith.addi %arg35, %c64_7 : index
                    %55 = arith.extsi %44 : vector<1x1x4x8xi8> to vector<1x1x4x8xi16>
                    %56 = arith.extsi %53 : vector<1x1x8x8xi8> to vector<1x1x8x8xi16>
                    %57 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %55, %56, %40 : vector<1x1x4x8xi16>, vector<1x1x8x8xi16> into vector<1x1x4x8xi32>
                    %58 = vector.transfer_read %collapse_shape_19[%arg36], %4 {in_bounds = [true]} : memref<4096xi8, 2 : i32>, vector<32xi8>
                    %59 = vector.shape_cast %58 : vector<32xi8> to vector<1x1x4x8xi8>
                    %60 = arith.addi %arg36, %c512_8 : index
                    %61 = arith.extsi %59 : vector<1x1x4x8xi8> to vector<1x1x4x8xi16>
                    %62 = arith.extsi %47 : vector<1x1x8x8xi8> to vector<1x1x8x8xi16>
                    %63 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %61, %62, %41 : vector<1x1x4x8xi16>, vector<1x1x8x8xi16> into vector<1x1x4x8xi32>
                    %64 = arith.extsi %59 : vector<1x1x4x8xi8> to vector<1x1x4x8xi16>
                    %65 = arith.extsi %53 : vector<1x1x8x8xi8> to vector<1x1x8x8xi16>
                    %66 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %64, %65, %42 : vector<1x1x4x8xi16>, vector<1x1x8x8xi16> into vector<1x1x4x8xi32>
                    %67 = vector.shape_cast %51 : vector<1x1x4x8xi32> to vector<32xi32>
                    %68 = vector.shape_cast %57 : vector<1x1x4x8xi32> to vector<32xi32>
                    %69 = vector.shape_cast %63 : vector<1x1x4x8xi32> to vector<32xi32>
                    %70 = vector.shape_cast %66 : vector<1x1x4x8xi32> to vector<32xi32>
                    scf.yield %67, %68, %69, %70, %45, %48, %54, %60 : vector<32xi32>, vector<32xi32>, vector<32xi32>, vector<32xi32>, index, index, index, index
                  }
                  %27 = arith.trunci %26#3 : vector<32xi32> to vector<32xi16>
                  %28 = arith.trunci %26#2 : vector<32xi32> to vector<32xi16>
                  %29 = arith.trunci %26#1 : vector<32xi32> to vector<32xi16>
                  %30 = arith.trunci %26#0 : vector<32xi32> to vector<32xi16>
                  %31 = vector.shape_cast %30 : vector<32xi16> to vector<1x1x4x8xi16>
                  %32 = vector.shape_cast %29 : vector<32xi16> to vector<1x1x4x8xi16>
                  %33 = vector.shape_cast %28 : vector<32xi16> to vector<1x1x4x8xi16>
                  %34 = vector.shape_cast %27 : vector<32xi16> to vector<1x1x4x8xi16>
                  %35 = affine.apply #map3()[%arg27]
                  %36 = affine.apply #map3()[%arg26]
                  vector.transfer_write %34, %arg22[%arg16, %arg17, %35, %36, %c0_14, %c0_14] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi16>, memref<4x4x16x16x4x8xi16, 2 : i32>
                  %37 = affine.apply #map3()[%arg26]
                  vector.transfer_write %33, %arg22[%arg16, %arg17, %arg27, %37, %c0_14, %c0_14] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi16>, memref<4x4x16x16x4x8xi16, 2 : i32>
                  %38 = affine.apply #map3()[%arg27]
                  vector.transfer_write %32, %arg22[%arg16, %arg17, %38, %arg26, %c0_14, %c0_14] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi16>, memref<4x4x16x16x4x8xi16, 2 : i32>
                  vector.transfer_write %31, %arg22[%arg16, %arg17, %arg27, %arg26, %c0_14, %c0_14] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xi16>, memref<4x4x16x16x4x8xi16, 2 : i32>
                }
              }
            }
          }
        }
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4) args(%arg19=%alloc_6, %arg20=%alloc_3) : memref<4x4x16x16x4x8xi16, 2 : i32>, memref<4x4x64x128xi16, 1 : i32> {
          %c512_7 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c4_8 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c8192_9 = arith.constant 8192 : index
          %c32768_10 = arith.constant 32768 : index
          %c128_11 = arith.constant 128 : index
          %c64_12 = arith.constant 64 : index
          %c1_13 = arith.constant 1 : index
          %c0_14 = arith.constant 0 : index
          air.dma_memcpy_nd (%arg20[%arg15, %arg16, %c0_14, %c0_14] [%c1_13, %c1_13, %c64_12, %c128_11] [%c32768_10, %c8192_9, %c128_11, %c1_13], %arg19[%arg15, %arg16, %c0_14, %c0_14, %c0_14, %c0_14] [%c1_13, %c1_13, %c16, %c4_8, %c16, %c8] [%c32768_10, %c8192_9, %c32, %c8, %c512_7, %c1_13]) : (memref<4x4x64x128xi16, 1 : i32>, memref<4x4x16x16x4x8xi16, 2 : i32>)
        }
        air.dma_memcpy_nd (%arg14[%0, %1] [%c256, %c512] [%c512, %c1_0], %alloc_3[%c0, %c0, %c0, %c0] [%c4, %c64, %c4, %c128] [%c32768, %c128, %c8192, %c1_0]) : (memref<512x512xi16>, memref<4x4x64x128xi16, 1 : i32>)
        memref.dealloc %alloc : memref<4x1x64x256xi8, 1 : i32>
        memref.dealloc %alloc_2 : memref<1x4x256x128xi8, 1 : i32>
        memref.dealloc %alloc_3 : memref<4x4x64x128xi16, 1 : i32>
        memref.dealloc %alloc_4 : memref<1x1x8x16x4x8xi8, 2 : i32>
        memref.dealloc %alloc_5 : memref<1x1x16x8x8x8xi8, 2 : i32>
        memref.dealloc %alloc_6 : memref<4x4x16x16x4x8xi16, 2 : i32>
      }
    }
    return
  }
}

