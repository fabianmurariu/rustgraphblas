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
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_FIRST_ $grb_tpe>], first);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_SECOND_ $grb_tpe>], second);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_MIN_ $grb_tpe>], min);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_MAX_ $grb_tpe>], max);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_PLUS_ $grb_tpe>], plus);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_MINUS_ $grb_tpe>], minus);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_TIMES_ $grb_tpe>], times);
                make_binary_op!($rust_tpe, $rust_tpe, [<GrB_DIV_ $grb_tpe>], div);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISEQ_ $grb_tpe>], is_eq);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISNE_ $grb_tpe>], is_neq);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISGT_ $grb_tpe>], is_gt);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISGE_ $grb_tpe>], is_gte);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISLT_ $grb_tpe>], is_lt);
                make_binary_op!($rust_tpe, $rust_tpe, [<GxB_ISLE_ $grb_tpe>], is_lte);
                )*
        }
    }
}

binary_ops_gen!(
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);



macro_rules! binary_ops_gen_bool{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_binary_op!($rust_tpe, bool, [<GrB_EQ_ $grb_tpe>], eq);
                make_binary_op!($rust_tpe, bool, [<GrB_NE_ $grb_tpe>], neq);
                make_binary_op!($rust_tpe, bool, [<GrB_GT_ $grb_tpe>], gt);
                make_binary_op!($rust_tpe, bool, [<GrB_LT_ $grb_tpe>], lt);
                make_binary_op!($rust_tpe, bool, [<GrB_GE_ $grb_tpe>], gte);
                make_binary_op!($rust_tpe, bool, [<GrB_LE_ $grb_tpe>], lte);
                )*
        }
    }
}

binary_ops_gen_bool!(
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);
