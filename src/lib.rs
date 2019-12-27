#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::mem::{MaybeUninit};
use enum_primitive::*;
use std::marker::PhantomData;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref GRB:u32 = unsafe {GrB_init(GrB_Mode_GrB_NONBLOCKING)};
}

enum_from_primitive! {
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u32)]
pub enum GrBIndex {
    Success = GrB_Info_GrB_SUCCESS,
    NoValue = GrB_Info_GrB_NO_VALUE,
}
}

pub struct SparseType {
    tpe: *mut GrB_Type,
}

pub trait TypeEncoder {
    fn blas_type() -> SparseType;
}

impl TypeEncoder for bool {
    fn blas_type() -> SparseType {
        let tpe = unsafe {&mut GrB_BOOL as *mut GrB_Type} ;
        SparseType{tpe: tpe}
    }
}

pub struct SparseMatrix<T>{
    mat: *mut GrB_Matrix,
    _marker: PhantomData<*const T>
}


impl<T:TypeEncoder> SparseMatrix<T> {
    pub fn new(size: (u64, u64)) -> SparseMatrix<T>{

        let _ = *GRB; // make sure lib is init ?
        let mut A = MaybeUninit::<GrB_Matrix>::uninit();

        let (rows, cols) = size;
        unsafe {
            GrB_Matrix_new(A.as_mut_ptr() , *T::blas_type().tpe, rows, cols);
        };

        let mat = A.as_mut_ptr();
        SparseMatrix{ mat: mat, _marker: PhantomData , }
    }

    pub fn rows(&mut self) -> u64 {
        let mut P = MaybeUninit::<u64>::uninit();

        unsafe {
            GrB_Matrix_nrows(P.as_mut_ptr(), *self.mat);
        }
        unsafe{P.assume_init()}
    }
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        unsafe { GrB_Matrix_free(self.mat);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bool_sparse_matrix() {
        let mut m = SparseMatrix::<bool>::new((5, 5));
        assert!(m.rows() == 5);
    }
}
