# ACACGS-rs

CS257 coursework re-written in Rust.
 
## Why?
- I wanted to play with [Rayon](https://github.com/rayon-rs/rayon) and intrinsics in Rust
- I wondered if it would be faster than C (spoiler: not quite yet)
- I wanted to try the new [`std::simd`](https://doc.rust-lang.org/nightly/std/simd/index.html) stuff but it ended up being an unnecessary (and bad in places) layer of abstraction considering I was targeting this purely for the i5-8500 on kudu
  - Might re-write to use this in future for portability so I can benchmark in on an M1 and a Ryzen too

## Usage
As usual, with cargo. If you want it to run on any other intel platform than an i5-8500 you might have to mess with `.cargo/config`. 

The `remoterun.sh` script has been modified to run this. It'll compile it for you, and then run `./remoterun.sh x y z` to run on batch compute.

## Performance
As it stands, it takes about 2.5 seconds for a 100x100x100 mesh, which is not that much slower than equivalent C code with a similar level of optimiation. If I get round to it, I might do some more detailed profiling and figure out where the performance difference lies.