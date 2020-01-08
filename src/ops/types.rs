#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::ops::ffi::*;

pub struct SparseType {
    pub(crate) tpe: *mut GrB_Type,
}

pub trait TypeEncoder {
    fn blas_type() -> SparseType;
}

#[macro_export]
macro_rules! make_base_sparse_type {
    ( $typ:ty, $grb_typ:ident ) => {
        impl TypeEncoder for $typ {
            fn blas_type() -> SparseType {
                let tpe = unsafe { &mut $grb_typ as *mut GrB_Type };
                SparseType { tpe: tpe }
            }
        }
    };
}

make_base_sparse_type!(bool, GrB_BOOL);
make_base_sparse_type!(i8, GrB_INT8);
make_base_sparse_type!(u8, GrB_UINT8);
make_base_sparse_type!(i16, GrB_INT16);
make_base_sparse_type!(u16, GrB_UINT16);
make_base_sparse_type!(i32, GrB_INT32);
make_base_sparse_type!(u32, GrB_UINT32);
make_base_sparse_type!(i64, GrB_INT64);
make_base_sparse_type!(u64, GrB_UINT64);

pub struct Descriptor {
    pub(crate) des: GrB_Descriptor
}