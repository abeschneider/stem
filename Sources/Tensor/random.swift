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

open class RandomNumberGenerator {
    var mt:[UInt64]
    var mti:Int
    
    init(seed:UInt64?=nil) {
        mt = Array<UInt64>(repeating: 0, count: N)
        mti = 0
        if let s = seed {
            setSeed(s)
        } else {
            // user timer to seed
//            setSeed(UInt32(mach_absolute_time()))
            setSeed(0)
        }
    }
    
    func setSeed(_ seed:UInt64) {
        mt[0] = seed & UInt64(0xffffffff)

        let m1:UInt64 = 1812433253
//        for (mti=1; mti<N; mti += 1) {
        mti = 1
        for _ in 1..<N {
            mt[mti] = (m1 * (mt[mti-1] ^ (mt[mti-1] >> 30)) + UInt64(mti))
            
            /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
            /* In the previous versions, MSBs of the seed affect   */
            /* only MSBs of the array mt[].                        */
            /* 2002/01/09 modified by Makoto Matsumoto             */
            mt[mti] &= 0xffffffff
            /* for >32 bit machines */
            
            mti += 1
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
        
        var y = mt[mti]
        mti += 1
        
        // tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18);
        
        return y
    }
}

var globalRng = RandomNumberGenerator()

/*
TODO:
bool: 	return [self randomUInt32] < 2147483648;
int32: 	return start + (floor([self randomDouble0To1Exclusive] * (double)width))
*/

extension Tensor where StorageType.ElementType:FloatNumericType {
    public func uniform(_ rng:RandomNumberGenerator=globalRng) {
        let prob:StorageType.ElementType = StorageType.ElementType(1.0/4294967295.0)
        for i in indices() {
            self[i] = StorageType.ElementType(rng.next())*prob
        }
    }
}

public func uniform<S:Storage>(_ shape:Extent) -> Tensor<S> where S.ElementType:FloatNumericType {
    let tensor = Tensor<S>(shape)
    tensor.uniform(globalRng)    
    return tensor
}

public func uniform() -> UInt64 {
    return globalRng.next()
}

public func uniform() -> Int {
    return Int(globalRng.next())
}

public func randInt(range:Range<Int>) -> Int {
    let width = UInt64(range.upperBound - range.lowerBound)
    let value = Int(globalRng.next() % width) + range.lowerBound
    return value
}

public func randInt(range:ClosedRange<Int>) -> Int {
    let width = UInt64(range.upperBound - range.lowerBound)
    let value = Int(globalRng.next() % width) + range.lowerBound
    return value
}

extension Array where Element:ExpressibleByIntegerLiteral {
    // inside-out algorithm (in-place Fisher-Yates)
    public func shuffle() -> Array<Element> {
        var result = Array<Element>(repeating: 0, count: self.count)
        for i in 0..<self.count-1 {
            let j = randInt(range: 0...i)
            if j != i { result[i] = result[j] }

            result[i] = self[i]
        }
        
        return result
    }
    
    // Fisher-Yates shuffle
    public mutating func shuffled() {
        for i in 0..<self.count-2 {
            let j = randInt(range: i..<self.count)
            swap(&self[i], &self[j])
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
