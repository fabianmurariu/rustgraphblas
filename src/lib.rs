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

// impl SparseType {
//     fn boolean() -> SparseType {

//         let _ = GRB;
//         let mut T = MaybeUninit::<GrB_Type>::new(GrB_BOOL);
//         let tpe = T.as_mut_ptr();
//         SparseType{tpe: tpe}

//     }
// }

pub struct SparseMatrix<T>{
    mat: *mut GrB_Matrix,
    phantom: PhantomData<*const T>
}

// pub trait HasSparseType {
//     fn sparseType() -> SparseType;
// }

// impl HasSparseType for bool {
//     fn sparseType() -> SparseType {
//         SparseType{ tpe: GrB_BOOL }
//     }
// }


impl<T> SparseMatrix<T> {
    pub fn new(size: (u64, u64)) -> SparseMatrix<T>{

        let _ = GRB; // make sure lib is init ?
        let mut A = MaybeUninit::<GrB_Matrix>::uninit();

        let (rows, cols) = size;
        unsafe {
            GrB_Matrix_new(A.as_mut_ptr() , GrB_BOOL, rows, cols);
        };

        let mat = A.as_mut_ptr();
        SparseMatrix{ mat: mat, phantom: PhantomData, }
    }
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        println!("BEFORE MATRIX FREE");
        unsafe { GrB_Matrix_free(self.mat);}
        println!("AFTER MATRIX FREE");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{self, MaybeUninit};

    #[test]
    fn create_bool_sparse_matrix() {
        SparseMatrix::<bool>::new((5, 5));
    }
}
