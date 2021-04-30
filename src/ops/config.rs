pub mod global {
use crate::ops::ffi::*;
use num_traits::ToPrimitive;


    #[derive(Clone, Copy, Debug, Eq, PartialEq, Primitive)]
    pub enum Field {
        Hyper = GxB_Option_Field_GxB_HYPER_SWITCH as isize,
        Bitmap = GxB_Option_Field_GxB_BITMAP_SWITCH as isize,
        Format = GxB_Option_Field_GxB_FORMAT as isize,
        Threads = GxB_Option_Field_GxB_GLOBAL_NTHREADS as isize,
        Chunk = GxB_Option_Field_GxB_GLOBAL_CHUNK as isize,
        Burble = GxB_Option_Field_GxB_BURBLE as isize,
        // Printf = GxB_Option_Field_GxB_PRINTF as isize,
        // Flush = GxB_Option_Field_GxB_FLUSH as isize,
        // MemoryPool = GxB_Option_Field_GxB_MEMORY_POOL as isize,
    }

    pub fn set_u32(f:Field, value: u32) {
        unsafe {GxB_Global_Option_set(f.to_u32().unwrap(), value) } ;
    }

    pub fn set_f64(f:Field, value: f64) {
        unsafe {GxB_Global_Option_set(f.to_u32().unwrap(), value) } ;
    }


}
