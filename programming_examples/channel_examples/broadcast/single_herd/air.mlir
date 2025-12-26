module {
  air.channel @ChanIn [1, 1] {broadcast_shape = [1 : index, 3 : index]}
  air.channel @ChanOut [1, 3]
  func.func @copy(%arg0: memref<6x8xi32>, %arg1: memref<6x8xi32>, %arg2: memref<6x8xi32>, %arg3: memref<6x8xi32>) {
    air.launch () in () args(%arg4=%arg0, %arg5=%arg1, %arg6=%arg2, %arg7=%arg3) : memref<6x8xi32>, memref<6x8xi32>, memref<6x8xi32>, memref<6x8xi32> {
      air.channel.put  @ChanIn[] (%arg4[] [] []) : (memref<6x8xi32>)
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      air.channel.get  @ChanOut[%c0, %c0_0] (%arg5[] [] []) : (memref<6x8xi32>)
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      air.channel.get  @ChanOut[%c0_1, %c1] (%arg6[] [] []) : (memref<6x8xi32>)
      %c0_2 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      air.channel.get  @ChanOut[%c0_2, %c2] (%arg7[] [] []) : (memref<6x8xi32>)
      air.segment @seg  {
        %c1_3 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        air.herd @broadcastherd  tile (%arg8, %arg9) in (%arg10=%c1_3, %arg11=%c3) {
          %alloc = memref.alloc() : memref<6x8xi32, 2 : i32>
          %alloc_4 = memref.alloc() : memref<6x8xi32, 2 : i32>
          air.channel.get  @ChanIn[%arg8, %arg9] (%alloc[] [] []) : (memref<6x8xi32, 2 : i32>)
          %c0_5 = arith.constant 0 : index
          %c6 = arith.constant 6 : index
          %c1_6 = arith.constant 1 : index
          scf.for %arg12 = %c0_5 to %c6 step %c1_6 {
            %c0_7 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg13 = %c0_7 to %c8 step %c1_8 {
              %0 = memref.load %alloc[%arg12, %arg13] : memref<6x8xi32, 2 : i32>
              %1 = arith.index_cast %arg9 : index to i32
              %2 = arith.addi %0, %1 : i32
              %c1_i32 = arith.constant 1 : i32
              %3 = arith.addi %2, %c1_i32 : i32
              memref.store %3, %alloc_4[%arg12, %arg13] : memref<6x8xi32, 2 : i32>
            }
          }
          air.channel.put  @ChanOut[%arg8, %arg9] (%alloc_4[] [] []) : (memref<6x8xi32, 2 : i32>)
          memref.dealloc %alloc : memref<6x8xi32, 2 : i32>
          memref.dealloc %alloc_4 : memref<6x8xi32, 2 : i32>
        }
      }
    }
    return
  }
}

