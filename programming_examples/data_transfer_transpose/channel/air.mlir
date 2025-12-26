module {
  air.channel @ChanIn []
  air.channel @ChanOut []
  func.func @transpose(%arg0: memref<64x32xui32>, %arg1: memref<32x64xui32>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<64x32xui32>, memref<32x64xui32> {
      air.channel.put  @ChanIn[] (%arg2[] [] []) : (memref<64x32xui32>)
      air.channel.get  @ChanOut[] (%arg3[] [] []) : (memref<32x64xui32>)
      air.segment @seg  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @herd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) {
          %alloc = memref.alloc() : memref<2048xui32, 2 : i32>
          air.channel.get  @ChanIn[] (%alloc[] [] []) : (memref<2048xui32, 2 : i32>)
          %c1_1 = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %c64 = arith.constant 64 : index
          %c1_2 = arith.constant 1 : index
          %c1_3 = arith.constant 1 : index
          %c32_4 = arith.constant 32 : index
          air.channel.put  @ChanOut[] (%alloc[] [%c1_1, %c32, %c64] [%c1_2, %c1_3, %c32_4]) : (memref<2048xui32, 2 : i32>)
          memref.dealloc %alloc : memref<2048xui32, 2 : i32>
        }
      }
    }
    return
  }
}

