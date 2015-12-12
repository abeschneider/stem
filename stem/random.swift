//
//  random.swift
//  stem
//
//  Created by Abe Schneider on 12/10/15.
//  Copyright © 2015 none. All rights reserved.
//

import Foundation

/* Based on MTRandom by Adam Preble. See: https://github.com/preble/MTRandom,
which in turn is based on mt19937ar.c; license is included directly below this comment block.

see:  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
also: Portions adapted from code by MrFusion: http://forums.macrumors.com/showthread.php?t=1083103


A C-program for MT19937, with initialization improved 2002/1/26.
Coded by Takuji Nishimura and Makoto Matsumoto.

Before using, initialize the state by using init_genrand(seed)
or init_by_array(init_key, key_length).

Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. The names of its contributors may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Any feedback is very welcome.
http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

// Period parameters
let N = 624
let M = 397
let MatrixA:UInt64 = 0x9908b0df     // constant vector a
let UpperMask:UInt64 = 0x80000000   // most significant w-r bits
let LowerMask:UInt64 = 0x7fffffff   // least significant r bits

class RandomNumberGenerator {
    var mt:[UInt64]
    var mti:Int
    
    init(seed:UInt64?=nil) {
        mt = Array<UInt64>(count: N, repeatedValue: 0)
        mti = 0
        if let s = seed {
            setSeed(s)
        } else {
            // user timer to seed
//            setSeed(UInt32(mach_absolute_time()))
            setSeed(0)
        }
    }
    
    func setSeed(seed:UInt64) {
        mt[0] = seed & UInt64(0xffffffff)

        let m1:UInt64 = 1812433253
        for (mti=1; mti<N; mti++) {
            mt[mti] = (m1 * (mt[mti-1] ^ (mt[mti-1] >> 30)) + UInt64(mti))
            
            /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
            /* In the previous versions, MSBs of the seed affect   */
            /* only MSBs of the array mt[].                        */
            /* 2002/01/09 modified by Makoto Matsumoto             */
            mt[mti] &= 0xffffffff
            /* for >32 bit machines */
        }
    }
    
    func next() -> UInt64 {
        // mag01[x] = x * MatriXA for x=0,1
        let mag01:[UInt64] = [0, MatrixA]
        
        // generate N words at one time
        if mti >= N {
            for k in 0..<(N-M) {
                let y:UInt64 = (mt[k] & UpperMask) | (mt[k+1] & LowerMask)
                mt[k] = mt[k+M] ^ (y >> 1) ^ mag01[Int(y & 0x1)]
            }
            
            for k in (N-M)..<N-1 {
                let y:UInt64 = (mt[k] & UpperMask) | (mt[k+1] & LowerMask)
                mt[k] = mt[k+(M-N)] ^ (y >> 1) ^ mag01[Int(y & 0x1)]
            }
            
            let y:UInt64 = (mt[N-1] & UpperMask) | (mt[0] & LowerMask)
            mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[Int(y & 0x1)]
            
            mti = 0
        }
        
        var y = mt[mti++]
        
        // tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18);
        
        return y
    }
}

/*
TODO:
bool: 	return [self randomUInt32] < 2147483648;
int32: 	return start + (floor([self randomDouble0To1Exclusive] * (double)width))
*/

extension Tensor where StorageType.ElementType == Float {
    func uniform(rng:RandomNumberGenerator) {
        for i in storageIndices() {
            // FIXME: this is probably incorrect
            storage[i] = Float(rng.next())*(1.0/4294967295.0)
        }
    }
}

extension Tensor where StorageType.ElementType == Double {
    func uniform(rng:RandomNumberGenerator) {
        for i in storageIndices() {
            storage[i] = Double(rng.next())*(1.0/4294967295.0)
        }
    }
}

//extension Tensor where StorageType.ElementType == Int {
//    func uniform(rng:RandomNumberGenerator, first:Int, last:Int, closed:Bool=false) {
//        var width = last - first
//        if closed { --width }
//        
//        for i in storageIndices() {
//            storage[i] = Int(rng.next()) % width + Int(first)
//        }
//    }
//}
