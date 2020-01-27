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

macro_rules! binary_ops_gen{
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

binary_ops_gen!(
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);
