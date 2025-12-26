module {
  func.func @copy(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4096xui8>, memref<4096xui8> {
      air.segment @seg  args(%arg4=%arg2, %arg5=%arg3) : memref<4096xui8>, memref<4096xui8> {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @copyherd  tile (%arg6, %arg7) in (%arg8=%c1, %arg9=%c1_0) args(%arg10=%arg4, %arg11=%arg5) : memref<4096xui8>, memref<4096xui8> {
          %c0 = arith.constant 0 : index
          %c4096 = arith.constant 4096 : index
          %c1024 = arith.constant 1024 : index
          scf.for %arg12 = %c0 to %c4096 step %c1024 {
            %alloc = memref.alloc() : memref<1024xui8, 2 : i32>
            %alloc_1 = memref.alloc() : memref<1024xui8, 2 : i32>
            %c1024_2 = arith.constant 1024 : index
            %c1_3 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc[] [] [], %arg10[%arg12] [%c1024_2] [%c1_3]) : (memref<1024xui8, 2 : i32>, memref<4096xui8>)
            %c0_4 = arith.constant 0 : index
            %c1024_5 = arith.constant 1024 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg13 = %c0_4 to %c1024_5 step %c1_6 {
              %0 = memref.load %alloc[%arg13] : memref<1024xui8, 2 : i32>
              memref.store %0, %alloc_1[%arg13] : memref<1024xui8, 2 : i32>
            }
            %c1024_7 = arith.constant 1024 : index
            %c1_8 = arith.constant 1 : index
            air.dma_memcpy_nd (%arg11[%arg12] [%c1024_7] [%c1_8], %alloc_1[] [] []) : (memref<4096xui8>, memref<1024xui8, 2 : i32>)
            memref.dealloc %alloc : memref<1024xui8, 2 : i32>
            memref.dealloc %alloc_1 : memref<1024xui8, 2 : i32>
          }
        }
      }
    }
    return
  }
}

