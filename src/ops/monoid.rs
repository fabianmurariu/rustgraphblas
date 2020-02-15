#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use crate::ops::binops::*;
use crate::ops::ffi::*;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

pub struct SparseMonoid<T> {
    pub(crate) m: GrB_Monoid,
    _t: PhantomData<*const T>,
}

impl<T: MonoidBuilder<T>> SparseMonoid<T> {
    pub fn new(binOp: BinaryOp<T, T, T>, default: T) -> SparseMonoid<T> {
        T::new_monoid(binOp, default)
    }
}

pub trait MonoidBuilder<T> {
    fn new_monoid(binOp: BinaryOp<T, T, T>, default: T) -> SparseMonoid<T>;
}

impl<T> Drop for SparseMonoid<T> {
    fn drop(&mut self) {
        unsafe {
            let m_pointer = &mut self.m as *mut GrB_Monoid;
            GrB_Monoid_free(m_pointer);
        }
    }
}

#[macro_export]
macro_rules! make_monoid_builder {
    ( $typ:ty, $builder:ident ) => {
        impl MonoidBuilder<$typ> for $typ {
            fn new_monoid(binOp: BinaryOp<$typ, $typ, $typ>, default: $typ) -> SparseMonoid<$typ> {
                let m = grb_call(|M:&mut MaybeUninit::<GrB_Monoid>|{
                    unsafe {$builder(M.as_mut_ptr(), binOp.op, default)}
                });

                SparseMonoid {m, _t: PhantomData}
            }
        }
    };
}

make_monoid_builder!(bool, GrB_Monoid_new_BOOL);
make_monoid_builder!(i8, GrB_Monoid_new_INT8);
make_monoid_builder!(u8, GrB_Monoid_new_UINT8);
make_monoid_builder!(i16, GrB_Monoid_new_INT16);
make_monoid_builder!(u16, GrB_Monoid_new_UINT16);
make_monoid_builder!(i32, GrB_Monoid_new_INT32);
make_monoid_builder!(u32, GrB_Monoid_new_UINT32);
make_monoid_builder!(i64, GrB_Monoid_new_INT64);
make_monoid_builder!(u64, GrB_Monoid_new_UINT64);
make_monoid_builder!(f32, GrB_Monoid_new_FP32);
make_monoid_builder!(f64, GrB_Monoid_new_FP64);

pub struct Semiring<'a, A, B, C> {
    _monoid: &'a SparseMonoid<C>,
    pub(crate) s: GrB_Semiring,
    _a: PhantomData<*const A>,
    _b: PhantomData<*const B>,
    _c: PhantomData<*const C>,
}

impl<'a, A, B, C> Semiring<'a, A, B, C> {
    pub fn new(add: &'a SparseMonoid<C>, multiply: BinaryOp<A, B, C>) -> Semiring<'a, A, B, C> {
        let s = grb_call(|S: &mut MaybeUninit<GrB_Semiring>| unsafe {
            GrB_Semiring_new(S.as_mut_ptr(), add.m, multiply.op)
        });

        Semiring {
            _monoid: add,
            s,
            _a: PhantomData,
            _b: PhantomData,
            _c: PhantomData,
        }
    }
}

impl<'a, A, B, C> Drop for Semiring<'a, A, B, C> {
    fn drop(&mut self) {
        unsafe {
            let m_pointer = &mut self.s as *mut GrB_Semiring;
            GrB_Semiring_free(m_pointer);
        }
    }
}

#[test]
fn create_semiring_bool_i32() {
    let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
    let land = BinaryOp::<bool, bool, bool>::land();
    Semiring::new(&m, land);
}
