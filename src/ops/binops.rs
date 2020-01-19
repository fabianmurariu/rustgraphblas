#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::ops::ffi::*;
use std::marker::PhantomData;

pub struct BinaryOp<A, B, C> {
    pub op: GrB_BinaryOp,
    _a: PhantomData<*const A>,
    _b: PhantomData<*const B>,
    _c: PhantomData<*const C>,
}

#[macro_export]
macro_rules! make_binary_op {
    ( $typ1:ty, $typ2:ty, $grb_op:ident, $op_name: ident) => {
        impl BinaryOp<$typ1, $typ1, $typ2> {
            pub fn $op_name() -> BinaryOp<$typ1, $typ1, $typ2> {
                unsafe {
                    BinaryOp {
                        op: $grb_op,
                        _a: PhantomData,
                        _b: PhantomData,
                        _c: PhantomData,
                    }
                }
            }
        }
    };
}

macro_rules! pairs{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_LAND_ $grb_tpe>], land);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_LOR_ $grb_tpe>], lor);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_LXOR_ $grb_tpe>], lxor);
                )*
        }
    }
}

pairs!(bool, i8, u8, i16, u16; BOOL, INT8, UINT8, INT16, UINT16);

make_binary_op!(i32, i32, GxB_LAND_INT32, land);
make_binary_op!(i32, i32, GxB_LOR_INT32, lor);
make_binary_op!(i32, i32, GxB_LXOR_INT32, lxor);

make_binary_op!(u32, u32, GxB_LAND_UINT32, land);
make_binary_op!(u32, u32, GxB_LOR_UINT32, lor);
make_binary_op!(u32, u32, GxB_LXOR_UINT32, lxor);

make_binary_op!(i64, i64, GxB_LAND_INT64, land);
make_binary_op!(i64, i64, GxB_LOR_INT64, lor);
make_binary_op!(i64, i64, GxB_LXOR_INT64, lxor);

make_binary_op!(u64, u64, GxB_LAND_UINT64, land);
make_binary_op!(u64, u64, GxB_LOR_UINT64, lor);
make_binary_op!(u64, u64, GxB_LXOR_UINT64, lxor);

make_binary_op!(f32, f32, GxB_LAND_FP32, land);
make_binary_op!(f32, f32, GxB_LOR_FP32, lor);
make_binary_op!(f32, f32, GxB_LXOR_FP32, lxor);

make_binary_op!(f64, f64, GxB_LAND_FP64, land);
make_binary_op!(f64, f64, GxB_LOR_FP64, lor);
make_binary_op!(f64, f64, GxB_LXOR_FP64, lxor);
