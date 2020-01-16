#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[macro_use]
extern crate enum_primitive_derive;
extern crate num_traits;

mod ops;
pub use crate::ops::binops::*;
pub use crate::ops::ffi::*;
pub use crate::ops::monoid::*;
pub use crate::ops::types::desc::*;
pub use crate::ops::types::*;

use enum_primitive::*;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref GRB: u32 = unsafe { GrB_init(GrB_Mode_GrB_NONBLOCKING) };
}

enum_from_primitive! {
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u32)]
pub enum GrBIndex {
    Success = GrB_Info_GrB_SUCCESS,
    NoValue = GrB_Info_GrB_NO_VALUE,
}
}

pub struct SparseMatrix<T> {
    mat: GrB_Matrix,
    _marker: PhantomData<*const T>,
}

pub struct SparseVector<T> {
    vec: GrB_Vector,
    _marker: PhantomData<*const T>,
}

impl<T: TypeEncoder> SparseMatrix<T> {
    pub fn empty(size: (u64, u64)) -> SparseMatrix<T> {
        let _ = *GRB;

        let mut A = MaybeUninit::<GrB_Matrix>::uninit();

        let (rows, cols) = size;
        unsafe {
            match GrB_Matrix_new(A.as_mut_ptr(), *T::blas_type().tpe, rows, cols) {
                0 => {
                    let mat = A.as_mut_ptr();
                    SparseMatrix {
                        mat: *mat,
                        _marker: PhantomData,
                    }
                }
                e => panic!("Failed to init matrix GrB_error {}", e),
            }
        }
    }

    pub fn rows(&mut self) -> u64 {
        let mut P = MaybeUninit::<u64>::uninit();

        unsafe {
            GrB_Matrix_nrows(P.as_mut_ptr(), self.mat);
        }
        unsafe { P.assume_init() }
    }
}

impl<T: TypeEncoder> SparseVector<T> {
    pub fn empty(size: u64) -> SparseVector<T> {
        let _ = *GRB;

        let mut V = MaybeUninit::<GrB_Vector>::uninit();
        unsafe {
            match GrB_Vector_new(V.as_mut_ptr(), *T::blas_type().tpe, size) {
                0 => {
                    let vec = V.as_mut_ptr();
                    SparseVector {
                        vec: *vec,
                        _marker: PhantomData,
                    }
                }
                e => panic!("Failed to init vector GrB_error {}", e),
            }
        }
    }
}

pub trait MatrixLike {
    type Item;

    fn insert(&mut self, row: u64, col: u64, val: Self::Item);
    fn get(&mut self, i: u64, j: u64) -> Option<Self::Item>;
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.mat as *mut GrB_Matrix;
        unsafe {
            GrB_Matrix_free(m_pointer);
        }
    }
}

impl<T> Drop for SparseVector<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.vec as *mut GrB_Vector;
        unsafe {
            GrB_Vector_free(m_pointer);
        }
    }
}

pub trait VectorLike {
    type Item;
    fn insert(&mut self, i: u64, val: Self::Item);
    fn get(&mut self, i: u64) -> Option<Self::Item>;
}

make_vector_like!(
    bool,
    GrB_Vector_extractElement_BOOL,
    GrB_Vector_setElement_BOOL
);
make_vector_like!(
    i8,
    GrB_Vector_extractElement_INT8,
    GrB_Vector_setElement_INT8
);
make_vector_like!(
    u8,
    GrB_Vector_extractElement_UINT8,
    GrB_Vector_setElement_UINT8
);
make_vector_like!(
    i16,
    GrB_Vector_extractElement_INT16,
    GrB_Vector_setElement_INT16
);
make_vector_like!(
    u16,
    GrB_Vector_extractElement_UINT16,
    GrB_Vector_setElement_UINT16
);
make_vector_like!(
    i32,
    GrB_Vector_extractElement_INT32,
    GrB_Vector_setElement_INT32
);
make_vector_like!(
    u32,
    GrB_Vector_extractElement_UINT32,
    GrB_Vector_setElement_UINT32
);
make_vector_like!(
    i64,
    GrB_Vector_extractElement_INT64,
    GrB_Vector_setElement_INT64
);
make_vector_like!(
    u64,
    GrB_Vector_extractElement_UINT64,
    GrB_Vector_setElement_UINT64
);

make_matrix_like!(
    bool,
    GrB_Matrix_extractElement_BOOL,
    GrB_Matrix_setElement_BOOL
);
make_matrix_like!(
    i8,
    GrB_Matrix_extractElement_INT8,
    GrB_Matrix_setElement_INT8
);
make_matrix_like!(
    u8,
    GrB_Matrix_extractElement_UINT8,
    GrB_Matrix_setElement_UINT8
);
make_matrix_like!(
    i16,
    GrB_Matrix_extractElement_INT16,
    GrB_Matrix_setElement_INT16
);
make_matrix_like!(
    u16,
    GrB_Matrix_extractElement_UINT16,
    GrB_Matrix_setElement_UINT16
);
make_matrix_like!(
    i32,
    GrB_Matrix_extractElement_INT32,
    GrB_Matrix_setElement_INT32
);
make_matrix_like!(
    u32,
    GrB_Matrix_extractElement_UINT32,
    GrB_Matrix_setElement_UINT32
);
make_matrix_like!(
    i64,
    GrB_Matrix_extractElement_INT64,
    GrB_Matrix_setElement_INT64
);
make_matrix_like!(
    u64,
    GrB_Matrix_extractElement_UINT64,
    GrB_Matrix_setElement_UINT64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bool_sparse_matrix() {
        let mut m = SparseMatrix::<bool>::empty((5, 5));
        // assert!(m.rows() == 5);
        assert!(m.insert(0, 3, true) == ());
        assert!(m.insert(1, 3, true) == ());
        assert!(m.insert(2, 3, true) == ());
        assert!(m.insert(3, 3, true) == ());
        assert!(m.insert(4, 3, true) == ());
        assert!(m.get(1, 3) == Some(true));
        assert!(m.get(0, 0) == None);
        assert!(m.get(1, 3) == Some(true));
        assert!(m.get(2, 3) == Some(true));
        assert!(m.get(0, 3) == Some(true));
    }

    #[test]
    fn create_u64_sparse_matrix() {
        let mut m = SparseMatrix::<u64>::empty((2, 5));
        assert!(m.insert(0, 0, 12 as u64) == ());
        assert!(m.get(0, 0) == Some(12 as u64));
    }

    #[test]
    fn create_u64_sparse_vector() {
        let mut v = SparseVector::<u64>::empty(10 as u64);
        assert!(v.insert(0, 3 as u64) == ());
        assert!(v.get(0) == Some(3 as u64));
    }
}

#[macro_export]
macro_rules! make_matrix_like {
    ( $typ:ty, $get_elem_func:ident, $set_elem_func:ident ) => {
        impl MatrixLike for SparseMatrix<$typ> {
            type Item = $typ;

            fn insert(&mut self, row: u64, col: u64, val: Self::Item) {
                unsafe {
                    match $set_elem_func(self.mat, val, row, col) {
                        0 => (),
                        e => panic!(
                            "Failed to set element at ({}, {})={} GrB_error: {}",
                            row, col, val, e
                        ),
                    }
                }
            }

            fn get(&mut self, i: u64, j: u64) -> Option<Self::Item> {
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.mat, i, j) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}, {}) GrB_error: {}", i, j, e),
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! make_vector_like {
    ( $typ:ty, $get_elem_func:ident, $set_elem_func:ident ) => {
        impl VectorLike for SparseVector<$typ> {
            type Item = $typ;

            fn insert(&mut self, i: u64, val: Self::Item) {
                unsafe {
                    match $set_elem_func(self.vec, val, i) {
                        0 => (),
                        e => panic!("Failed to set element at ({})={} GrB_error: {}", i, val, e),
                    }
                }
            }

            fn get(&mut self, i: u64) -> Option<Self::Item> {
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.vec, i) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}) GrB_error: {}", i, e),
                    }
                }
            }
        }
    };
}

// finally we get some stuff done
trait VectorAlgebra<Z> {
    fn vxm<X, Y>(
        &mut self,
        m: &SparseMatrix<Y>,
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;
}

trait Reduce<T> {
    fn reduce<'m>(
        &self,
        init: &'m mut T,
        acc: Option<BinaryOp<T, T, T>>,
        monoid: SparseMonoid<T>,
        desc: Descriptor,
    ) -> &'m T;
}

impl Reduce<bool> for SparseVector<bool> {
    fn reduce<'m>(
        &self,
        init: &'m mut bool,
        acc: Option<BinaryOp<bool, bool, bool>>,
        monoid: SparseMonoid<bool>,
        desc: Descriptor,
    ) -> &'m bool {
        unsafe {
            match acc {
                Some(op) => {
                    GrB_Vector_reduce_BOOL(init, op.op, monoid.m, self.vec, desc.desc);
                },
                None => {
                    let m = ptr::null_mut::<GB_BinaryOp_opaque>();
                    GrB_Vector_reduce_BOOL(init, m, monoid.m, self.vec, desc.desc);
                }
            }
        }
        init
    }
}

impl <Z> VectorAlgebra<Z> for SparseVector<Z> {

    fn vxm<X, Y>(
        &mut self,
        m: &SparseMatrix<Y>,
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor
    ) -> &SparseVector<Z> {
        println!("VXM");

        let mask = ptr::null_mut::<GB_Vector_opaque>();
        let acc = ptr::null_mut::<GB_BinaryOp_opaque>();
        unsafe {
            match GrB_vxm(self.vec, mask, acc, s_ring.s, self.vec, m.mat, desc.desc){
                0 => self,
                err => panic!("VXM failed GrB_error {}", err)
            }
        }
    }
}

#[test]
fn vmx_bool() {
    let mut v = SparseVector::<bool>::empty(10);
    v.insert(0, true);
    v.insert(2, true);
    v.insert(4, true);

    let mut A = SparseMatrix::<bool>::empty((10, 10));
    A.insert(0, 0, true);
    A.insert(1, 0, true);
    A.insert(0, 1, true);

    let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
    let land = BinaryOp::<bool, bool, bool>::land();
    let semi = Semiring::new(m, land);

    v.vxm(&A, semi, &Descriptor::default());

}

#[test]
fn reduce_vector_and_all_true() {
    let mut v = SparseVector::<bool>::empty(2);
    v.insert(0, true);
    v.insert(1, true);

    let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), true);
    let land = BinaryOp::<bool, bool, bool>::land();
    let desc = Descriptor::default();

    assert_eq!(*v.reduce(&mut true, None, m, desc), true);
}

#[test]
fn reduce_vector_and_some_true() {
    let mut v = SparseVector::<bool>::empty(2);
    v.insert(0, true);

    let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), false);
    let land = BinaryOp::<bool, bool, bool>::land();
    let desc = Descriptor::default();

    assert_eq!(*v.reduce(&mut true, None, m, desc), false);
}
