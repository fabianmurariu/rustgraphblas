#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::GRB;
use crate::{ops::ffi::*, TypeEncoder};
use std::{marker::PhantomData, mem::MaybeUninit};

pub struct UnaryOp<X, Z> {
    pub op: GrB_UnaryOp,
    _x: PhantomData<*const X>,
    _z: PhantomData<*const Z>,
}

pub trait UnOp {
    type X;
    fn op(&mut self, x: &Self::X) -> ();
}

impl<X, Z> UnaryOp<X, Z>
where
    Z: UnOp<X = X>,
{
    #[no_mangle]
    extern "C" fn unsafe_call(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *const ::std::os::raw::c_void,
    ) -> () {
        let z: &mut Z = unsafe { &mut *(arg1 as *mut Z) };
        let x: &X = unsafe { &*(arg2 as *const X) };
        z.op(x)
    }

    pub fn new() -> Self
    where
        Z: TypeEncoder,
        X: TypeEncoder,
    {
        let _ = *GRB;
        let grb_op: GrB_UnaryOp = grb_call(|OP: &mut MaybeUninit<GrB_UnaryOp>| unsafe {
            GrB_UnaryOp_new(
                OP.as_mut_ptr(),
                Some(Self::unsafe_call),
                Z::blas_type().tpe,
                X::blas_type().tpe,
            )
        });
        UnaryOp {
            op: grb_op,
            _x: PhantomData,
            _z: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! make_unary_op {
    ( $typ1:ty, $typ2:ty, $grb_op:ident, $op_name: ident) => {
        impl UnaryOp<$typ1, $typ2> {
            pub fn $op_name() -> UnaryOp<$typ1, $typ2> {
                unsafe {
                    UnaryOp {
                        op: $grb_op,
                        _x: PhantomData,
                        _z: PhantomData,
                    }
                }
            }
        }
    };
}

macro_rules! unary_ops_gen_fp_fc{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_SQRT_ $grb_tpe>], sqrt);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_LOG_ $grb_tpe>], log);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_EXP_ $grb_tpe>],exp);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_LOG2_ $grb_tpe>], log2);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_SIN_ $grb_tpe>], sin);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_COS_ $grb_tpe>],cos);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_TAN_ $grb_tpe>], tan);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ACOS_ $grb_tpe>], acos);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ASIN_ $grb_tpe>], asin);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ATAN_ $grb_tpe>], atan);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_SINH_ $grb_tpe>], sinh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_COSH_ $grb_tpe>], cosh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_TANH_ $grb_tpe>], tanh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ACOSH_ $grb_tpe>], acosh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ASINH_ $grb_tpe>], asinh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ATANH_ $grb_tpe>], atanh);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_SIGNUM_ $grb_tpe>], signum);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_CEIL_ $grb_tpe>], ceil);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_FLOOR_ $grb_tpe>], floor);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ROUND_ $grb_tpe>], round);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_TRUNC_ $grb_tpe>], trunc);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_EXP2_ $grb_tpe>], exp2);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_EXPM1_ $grb_tpe>], expm1);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_LOG10_ $grb_tpe>], log10);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_LOG1P_ $grb_tpe>], log1p);
                )*
        }
    }
}

macro_rules! unary_ops_gen_fc{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_IDENTITY_ $grb_tpe>], identity);
                )*
        }
    }
}

macro_rules! unary_ops_gen_int{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_unary_op!($rust_tpe, $rust_tpe, [<GrB_BNOT_ $grb_tpe>], bnot);
                )*
        }
    }
}

macro_rules! unary_ops_gen_core{
    ( $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_unary_op!($rust_tpe, $rust_tpe, [<GrB_IDENTITY_ $grb_tpe>], identity);
                make_unary_op!($rust_tpe, $rust_tpe, [<GrB_AINV_ $grb_tpe>], ainv);
                make_unary_op!($rust_tpe, $rust_tpe, [<GrB_MINV_ $grb_tpe>], minv);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_LNOT_ $grb_tpe>], lnot);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ONE_ $grb_tpe>], one);
                make_unary_op!($rust_tpe, $rust_tpe, [<GxB_ABS_ $grb_tpe>], abs);
                )*
        }
    }
}

unary_ops_gen_fc!(
    Complex<f32>, Complex<f64>;
    FC32, FC64);

unary_ops_gen_fp_fc!(
    f32, f64, Complex<f32>, Complex<f64>;
    FP32, FP64,FC32, FC64);

unary_ops_gen_core!(
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

unary_ops_gen_int!(
    i8, u8, i16, u16, i32, u32, i64, u64;
    INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64);
