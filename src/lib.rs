#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

#[macro_use]
extern crate enum_primitive_derive;
extern crate num_traits;

pub mod ops;
pub mod algo;
pub use crate::ops::binops::*;
pub use crate::ops::ffi::*;
pub use crate::ops::monoid::*;
pub use crate::ops::types::desc::*;
pub use crate::ops::types::*;
pub use crate::ops::vector_algebra::*;
pub use crate::ops::matrix_algebra::*;

use std::ptr;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_traits::{FromPrimitive, ToPrimitive};
#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref GRB: u32 = unsafe { GrB_init(GrB_Mode_GrB_NONBLOCKING) };
}

pub struct SparseMatrix<T> {
    mat: GrB_Matrix,
    _marker: PhantomData<*const T>,
}

impl<T> fmt::Debug for SparseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.shape();
        let nvals = self.nvals();
        let format = self.get_format();
        write!(f, "SparseMatrix[shape=({}x{}), vals={}, format={}]", r, c, nvals, format.to_u32().unwrap())
    }
}

pub struct SparseVector<T> {
    vec: GrB_Vector,
    _marker: PhantomData<*const T>,
}

impl<T> fmt::Debug for SparseVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.size();
        let nvals = self.nvals();
        write!(f, "SparseVector[size={}, vals={}]", shape, nvals)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Primitive)]
pub enum MatrixFormat {
    CSR = GxB_Format_Value_GxB_BY_ROW as isize,
    CSC = GxB_Format_Value_GxB_BY_COL as isize,
}

impl<T: TypeEncoder> SparseMatrix<T> {
    pub fn empty_mat(size: (u64, u64), format: Option<MatrixFormat>) -> SparseMatrix<T> {
        let _ = *GRB;

        let (rows, cols) = size;

        let mat = grb_call(|M: &mut MaybeUninit<GrB_Matrix>| unsafe {
            GrB_Matrix_new(M.as_mut_ptr(), *T::blas_type().tpe, rows, cols)
        });

        if let Some(fmt) = format {
            grb_run(|| unsafe {
                GxB_Matrix_Option_set(mat, GxB_Option_Field_GxB_FORMAT, fmt.to_u32().unwrap())
            });
        }

        SparseMatrix {
            mat,
            _marker: PhantomData,
        }
    }

    pub fn empty(size: (u64, u64)) -> SparseMatrix<T> {
        SparseMatrix::<T>::empty_mat(size, None)
    }

    pub fn empty_csr(size: (u64, u64)) -> SparseMatrix<T> {
        SparseMatrix::<T>::empty_mat(size, Some(MatrixFormat::CSR))
    }

    pub fn empty_csc(size: (u64, u64)) -> SparseMatrix<T> {
        SparseMatrix::<T>::empty_mat(size, Some(MatrixFormat::CSC))
    }
}

impl<T> SparseMatrix<T> {
    pub fn rows(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_nrows(G.as_mut_ptr(), self.mat) })
    }

    pub fn cols(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_ncols(G.as_mut_ptr(), self.mat) })
    }

    pub fn shape(&self) -> (u64, u64) {
        (self.rows(), self.cols())
    }

    pub fn nvals(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_nvals(G.as_mut_ptr(), self.mat) })
    }

    pub fn get_format(&self) -> MatrixFormat {
        let grb_fmt = grb_call(|F: &mut MaybeUninit<u32>| unsafe {
            GxB_Matrix_Option_get(self.mat, GxB_Option_Field_GxB_FORMAT, F.as_mut_ptr())
        });
        MatrixFormat::from_u32(grb_fmt).unwrap()
    }
}

impl<T: TypeEncoder> SparseVector<T> {
    pub fn empty(size: u64) -> SparseVector<T> {
        let _ = *GRB;

        let vec = grb_call(|V: &mut MaybeUninit<GrB_Vector>| unsafe {
            GrB_Vector_new(V.as_mut_ptr(), *T::blas_type().tpe, size)
        });
        SparseVector {
            vec,
            _marker: PhantomData,
        }
    }
}

impl<T> SparseVector<T> {
    pub fn nvals(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Vector_nvals(G.as_mut_ptr(), self.vec) })
    }

    pub fn size(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Vector_size(G.as_mut_ptr(), self.vec) })
    }
}

pub trait MatrixLike {
    type Item;

    fn insert(&mut self, row: u64, col: u64, val: Self::Item);
    fn get(&self, i: u64, j: u64) -> Option<Self::Item>;
    fn resize(&mut self, new_row:u64, new_col:u64);
    fn transpose_mut(&mut self) -> &Self;
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.mat as *mut GrB_Matrix;
        grb_run({ || unsafe { GrB_Matrix_free(m_pointer) } });
    }
}

impl<T> Drop for SparseVector<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.vec as *mut GrB_Vector;
        //nval forces a local wait for all the pending computations on this vector
        // FIXME: should we really call this here? when using arrays in tight loops we need to hold on to them
        // and not trigger a wait
        // self.nvals();
        grb_run(|| unsafe { GrB_Vector_free(m_pointer) });
    }
}

pub trait VectorLike {
    type Item;
    fn insert(&mut self, i: u64, val: Self::Item);
    fn get(&self, i: u64) -> Option<Self::Item>;
    fn resize(&mut self, new_size:u64);
}

macro_rules! sparse_vector_tpe_gen{
    ( $grb_sparse_tpe:ident; $rust_maker:ident; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                $rust_maker!(
                    $rust_tpe,
                    [<$grb_sparse_tpe _extractElement_ $grb_tpe>],
                    [<$grb_sparse_tpe _setElement_ $grb_tpe>]
                );
                )*
        }
    }
}

sparse_vector_tpe_gen!(GrB_Vector; make_vector_like;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

sparse_vector_tpe_gen!(GrB_Matrix; make_matrix_like;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

#[macro_export]
macro_rules! make_matrix_like {
    ( $typ:ty, $get_elem_func:ident, $set_elem_func:ident ) => {
        impl MatrixLike for SparseMatrix<$typ> {
            type Item = $typ;

            fn insert(&mut self, row: u64, col: u64, val: Self::Item) {
                grb_run(|| unsafe { $set_elem_func(self.mat, val, row, col) })
            }

            fn get(&self, i: u64, j: u64) -> Option<Self::Item> {
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.mat, i, j) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}, {}) GrB_error: {}", i, j, e),
                    }
                }
            }

            fn resize(&mut self, new_row:u64, new_col:u64) {
                grb_run(|| unsafe { GxB_Matrix_resize(self.mat, new_row, new_col) })
            }

            fn transpose_mut(&mut self) -> &Self {
                grb_run(|| unsafe {
                    GrB_transpose(self.mat, ptr::null_mut::<GB_Matrix_opaque>(), ptr::null_mut::<GB_BinaryOp_opaque>(), self.mat, Descriptor::new().set(Field::Input0, Value::Transpose).desc)
                });
                self
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
                grb_run(|| unsafe { $set_elem_func(self.vec, val, i) })
            }

            fn get(&self, i: u64) -> Option<Self::Item> {
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.vec, i) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}) GrB_error: {}", i, e),
                    }
                }
            }

            fn resize(&mut self, new_size:u64) {
                grb_run(|| unsafe { GxB_Vector_resize(self.vec, new_size) })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bool_sparse_matrix() {
        let mut m = SparseMatrix::<bool>::empty((5, 5));
        assert!(m.rows() == 5);
        assert!(m.insert(0, 3, true) == ());
        assert!(m.insert(1, 3, true) == ());
        assert!(m.insert(2, 3, true) == ());
        assert!(m.insert(3, 3, true) == ());
        assert!(m.insert(4, 3, true) == ());
        assert!(m.nvals() == 5);
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

    #[test]
    fn vmx_bool() {
        let mut v = SparseVector::<bool>::empty(10);
        v.load(3, &[true, true, true], &[0, 4, 2]);

        let mut A = SparseMatrix::<bool>::empty((10, 10));
        A.insert(0, 0, true);
        A.insert(1, 0, true);
        A.insert(0, 1, true);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let land = BinaryOp::<bool, bool, bool>::land();
        let semi = Semiring::new(&m, land);

        v.vxm(empty_mask::<bool>(), None, &A, &semi, &Descriptor::default());
    }

    #[test]
    fn reduce_vector_and_all_true() {
        let mut v = SparseVector::<bool>::empty(2);
        v.insert(0, true);
        v.insert(1, true);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), true);
        let desc = Descriptor::default();

        assert_eq!(*v.reduce(&mut true, None, &m, &desc), true);
    }

    #[test]
    fn reduce_vector_and_some_true_some_false() {
        let mut v = SparseVector::<bool>::empty(2);
        v.insert(0, true);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), false);
        let desc = Descriptor::default();

        assert_eq!(*v.reduce(&mut true, None, &m, &desc), false);
    }

    #[test]
    fn create_matrix_with_csc_and_csr_format() {
        let A = SparseMatrix::<bool>::empty_csr((2, 3));
        let B = SparseMatrix::<bool>::empty_csc((4, 5));
        let C = SparseMatrix::<bool>::empty((4, 5));

        assert_eq!(A.get_format(), MatrixFormat::CSR);
        assert_eq!(B.get_format(), MatrixFormat::CSC);
        assert_eq!(C.get_format(), MatrixFormat::CSR); // DEFAULT
    }

    #[test]
    /**
     * this is the test for the graph on the cover of
     * Graph Algorithms in the Language of Linear Algebra
     * where by multiplying a boolean matrix with
     * a boolean vector on the and/or semiring we effectively find the neighbours
     * for the input vertex
     * */
    fn define_graph_adj_matrix_one_hop_neighbours() {
        let mut m = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        m.load(
            edges_n as u64,
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        let mut v = SparseVector::<bool>::empty(7);
        v.insert(0, true);

        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());
        v.vxm(
            empty_mask::<bool>(),
            None,
            &m,
            &or_and_semi,
            &Descriptor::default(),
        );

        // neighbours of 0 are 1 and 3
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(3), Some(true));

        // the rest is set to null
        assert_eq!(v.get(0), None);
        assert_eq!(v.get(2), None);
        assert_eq!(v.get(4), None);
        assert_eq!(v.get(5), None);
        assert_eq!(v.get(6), None);
    }

    #[test]
    fn resize_changes_the_shape_of_a_vector(){
        let mut v = SparseVector::<bool>::empty(5);

        let s = v.size();
        assert_eq!(s, 5);

        v.resize(10);

        let s = v.size();
        assert_eq!(s, 10);
    }

    #[test]
    fn resize_changes_the_shape_of_a_matrix(){
        let mut v = SparseMatrix::<bool>::empty((5, 7));

        let (r, c) = v.shape();
        assert_eq!(r, 5);
        assert_eq!(c, 7);

        v.resize(9, 8);

        let (r, c) = v.shape();
        assert_eq!(r, 9);
        assert_eq!(c, 8);
    }

    #[test]
    fn transpose_flips_the_shape_of_the_matrix(){
        let mut v = SparseMatrix::<bool>::empty((5, 7));

        let (r, c) = v.shape();
        assert_eq!(r, 5);
        assert_eq!(c, 7);

        v.transpose_mut();

        let (r, c) = v.shape();
        assert_eq!(r, 5);
        assert_eq!(c, 7);
    }
}
