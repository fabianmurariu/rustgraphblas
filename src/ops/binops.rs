#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::GRB;
use crate::{ops::ffi::*, TypeEncoder};
use std::{marker::PhantomData, mem::MaybeUninit};
pub struct BinaryOp<A, B, C> {
    pub op: GrB_BinaryOp,
    _a: PhantomData<*const A>,
    _b: PhantomData<*const B>,
    _c: PhantomData<*const C>,
}

pub trait BinOp {
    type X;
    type Y;
    fn op(&mut self, x: &Self::X, y: &Self::Y) -> ();
}

impl<A, B, C> BinaryOp<A, B, C>
where
    C: BinOp<X = A, Y = B>,
{
    #[no_mangle]
    extern "C" fn unsafe_call(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *const ::std::os::raw::c_void,
        arg3: *const ::std::os::raw::c_void,
    ) -> () {
        let z: &mut C = unsafe { &mut *(arg1 as *mut C) };
        let x: &A = unsafe { &*(arg2 as *const A) };
        let y: &B = unsafe { &*(arg3 as *const B) };
        z.op(x, y)
    }

    pub fn new() -> Self
    where
        A: TypeEncoder,
        B: TypeEncoder,
        C: TypeEncoder,
    {
        let _ = *GRB;
        let grb_op: GrB_BinaryOp = grb_call(|OP: &mut MaybeUninit<GrB_BinaryOp>| unsafe {
            GrB_BinaryOp_new(
                OP.as_mut_ptr(),
                Some(Self::unsafe_call),
                C::blas_type().tpe,
                A::blas_type().tpe,
                B::blas_type().tpe,
            )
        });
        BinaryOp {
            op: grb_op,
            _a: PhantomData,
            _b: PhantomData,
            _c: PhantomData,
        }
    }
}


impl<A> BinaryOp<A, A, A>
where
    A: BinOp<X = A, Y = A>,
{

    pub fn combine() -> Self
    where
        A: TypeEncoder
    {
        let _ = *GRB;
        let grb_op: GrB_BinaryOp = grb_call(|OP: &mut MaybeUninit<GrB_BinaryOp>| unsafe {
            let tpe = A::blas_type().tpe;
            GrB_BinaryOp_new(
                OP.as_mut_ptr(),
                Some(Self::unsafe_call),
                tpe,
                tpe,
                tpe
            )
        });
        BinaryOp {
            op: grb_op,
            _a: PhantomData,
            _b: PhantomData,
            _c: PhantomData,
        }
    }
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
    ( GrB; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
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
    };
    ( GxB; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                make_binary_op!($rust_tpe, bool, [<GxB_EQ_ $grb_tpe>], eq);
                make_binary_op!($rust_tpe, bool, [<GxB_NE_ $grb_tpe>], neq);
                )*
        }
    }
}

binary_ops_gen_bool!(GrB;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

binary_ops_gen_bool!(GxB; Complex<f32>, Complex<f64>; FC32, FC64);

#[cfg(test)]
mod tests {

    use super::*;
    use crate::*;

    #[repr(C)]
    #[derive(Debug, PartialEq)]
    struct Complex {
        r: f64,
        i: f64,
    }

    impl BinOp for Id<Complex> {
        type X = Id<Complex>;
        type Y = Id<Complex>;

        fn op(&mut self, x: &Self::X, y: &Self::Y) -> () {
            self.0.r = x.0.r + y.0.r;
            self.0.i = x.0.i + y.0.i;
        }
    }

    #[test]
    fn createBinaryOpFromRustType() {
        let mut mat = SparseMatrix::<Id<Complex>>::empty((12, 12));
        let op = BinaryOp::<Id<Complex>, Id<Complex>, Id<Complex>>::combine();
        let _m = SparseMonoid::<Id<Complex>>::new(op, Id(Complex{r: 0.0, i: 0.0}));


        mat.insert((0, 0), Id(Complex { i: 0.4, r: 0.9 }));
        let expected = Id(Complex { i: 0.4, r: 0.9 });
        if let Some(actual) = mat.get((0, 0)) {
            assert_eq!(actual, expected)
        } else {
            panic!("Unable to get custom type from GrB_Matrix")
        }
    }
}
