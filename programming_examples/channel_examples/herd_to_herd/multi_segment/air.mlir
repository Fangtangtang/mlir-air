module {
  air.channel @ChanIn []
  air.channel @ChanOut []
  air.channel @Herd2Herd []
  func.func @copy(%arg0: memref<16x32xi32>, %arg1: memref<16x32xi32>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<16x32xi32>, memref<16x32xi32> {
      air.channel.put  @ChanIn[] (%arg2[] [] []) : (memref<16x32xi32>)
      air.channel.get  @ChanOut[] (%arg3[] [] []) : (memref<16x32xi32>)
      air.segment @producer_segment  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @producer_herd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) {
          %alloc = memref.alloc() : memref<16x32xi32, 2 : i32>
          %alloc_1 = memref.alloc() : memref<16x32xi32, 2 : i32>
          air.channel.get  @ChanIn[] (%alloc[] [] []) : (memref<16x32xi32, 2 : i32>)
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %arg8 = %c0 to %c16 step %c1_2 {
            %c0_3 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c1_4 = arith.constant 1 : index
            scf.for %arg9 = %c0_3 to %c32 step %c1_4 {
              %0 = memref.load %alloc[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
              %1 = arith.muli %0, %0 : i32
              memref.store %1, %alloc_1[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
            }
          }
          air.channel.put  @Herd2Herd[] (%alloc_1[] [] []) : (memref<16x32xi32, 2 : i32>)
          memref.dealloc %alloc : memref<16x32xi32, 2 : i32>
          memref.dealloc %alloc_1 : memref<16x32xi32, 2 : i32>
        }
      }
      air.segment @consumer_segment  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @consumer_herd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) {
          %alloc = memref.alloc() : memref<16x32xi32, 2 : i32>
          %alloc_1 = memref.alloc() : memref<16x32xi32, 2 : i32>
          air.channel.get  @Herd2Herd[] (%alloc[] [] []) : (memref<16x32xi32, 2 : i32>)
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %arg8 = %c0 to %c16 step %c1_2 {
            %c0_3 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c1_4 = arith.constant 1 : index
            scf.for %arg9 = %c0_3 to %c32 step %c1_4 {
              %0 = memref.load %alloc[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
              %c1_i32 = arith.constant 1 : i32
              %1 = arith.addi %0, %c1_i32 : i32
              memref.store %1, %alloc_1[%arg8, %arg9] : memref<16x32xi32, 2 : i32>
            }
          }
          air.channel.put  @ChanOut[] (%alloc_1[] [] []) : (memref<16x32xi32, 2 : i32>)
          memref.dealloc %alloc : memref<16x32xi32, 2 : i32>
          memref.dealloc %alloc_1 : memref<16x32xi32, 2 : i32>
        }
      }
    }
    return
  }
}

