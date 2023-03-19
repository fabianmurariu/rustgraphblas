#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

#[macro_use]
extern crate enum_primitive_derive;
extern crate num_traits;

pub mod algo;
pub mod ops;
pub use crate::ops::binops::*;
pub use crate::ops::config::*;
pub use crate::ops::elem_wise::*;
use crate::ops::ffi::*;
pub use crate::ops::index::*;
pub use crate::ops::matrix_algebra::*;
pub use crate::ops::monoid::*;
pub use crate::ops::reduce::*;
pub use crate::ops::types::desc::*;
pub use crate::ops::types::*;
pub use crate::ops::vector_algebra::*;

use num_traits::{FromPrimitive, ToPrimitive};
use ops::unaryops::UnaryOp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref GRB: i32 = unsafe { GrB_init(GrB_Mode_GrB_NONBLOCKING) };
}

type GrBInfo = i32;

// Wrapper for custom types to allow calling different traits for UDFs
#[derive(Debug, PartialEq)]
struct Udf<T>(T);

pub struct SparseMatrix<T> {
    inner: GrB_Matrix,
    _marker: PhantomData<*const T>,
}

unsafe impl<T: Send> Send for SparseMatrix<T> {}
unsafe impl<T: Send> Send for SparseVector<T> {}

pub struct SparseVector<T> {
    inner: GrB_Vector,
    _marker: PhantomData<*const T>,
}

pub struct SyncSparseMatrix<T> {
    m: Arc<Mutex<SparseMatrix<T>>>,
}

pub struct SyncSparseVector<T> {
    m: Arc<Mutex<SparseVector<T>>>,
}

unsafe impl<T: Sync> Sync for SyncSparseMatrix<T> {}
unsafe impl<T: Sync> Sync for SyncSparseVector<T> {}

impl<T> Clone for SyncSparseMatrix<T> {
    fn clone(&self) -> Self {
        Self { m: self.m.clone() }
    }
}

impl<T> Clone for SyncSparseVector<T> {
    fn clone(&self) -> Self {
        Self { m: self.m.clone() }
    }
}

impl<T> SyncSparseMatrix<T> {
    pub fn from_mat(mat: SparseMatrix<T>) -> Self {
        SyncSparseMatrix {
            m: Arc::new(Mutex::new(mat)),
        }
    }

    pub fn use_mut<F, B>(&mut self, mut sync_fn: F) -> B
    where
        F: FnMut(&mut SparseMatrix<T>) -> B,
    {
        let mut grb_mat = self.m.lock().unwrap();
        sync_fn(&mut grb_mat)
    }

    pub fn use0<F, B>(&self, sync_fn: F) -> B
    where
        F: Fn(&SparseMatrix<T>) -> B,
    {
        let grb_mat = self.m.lock().unwrap();
        sync_fn(&grb_mat)
    }
}

impl<T> SyncSparseVector<T> {
    pub fn from_mat(mat: SparseVector<T>) -> Self {
        SyncSparseVector {
            m: Arc::new(Mutex::new(mat)),
        }
    }

    pub fn use_mut<F, B>(&mut self, mut sync_fn: F) -> B
    where
        F: FnMut(&mut SparseVector<T>) -> B,
    {
        let mut grb_mat = self.m.lock().unwrap();
        sync_fn(&mut grb_mat)
    }

    pub fn use0<F, B>(&self, sync_fn: F) -> B
    where
        F: Fn(&SparseVector<T>) -> B,
    {
        let grb_mat = self.m.lock().unwrap();
        sync_fn(&grb_mat)
    }
}

impl<T> fmt::Debug for SparseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.shape();
        let nvals = self.nvals();
        let format = self.get_format();
        write!(
            f,
            "SparseMatrix[shape=({}x{}), vals={}, format={}]",
            r,
            c,
            nvals,
            format.to_u32().unwrap()
        )
    }
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
            GrB_Matrix_new(M.as_mut_ptr(), T::blas_type().tpe, rows, cols)
        });

        if let Some(fmt) = format {
            grb_run(|| unsafe {
                GxB_Matrix_Option_set(mat, GxB_Option_Field_GxB_FORMAT, fmt.to_u32().unwrap())
            });
        }

        SparseMatrix {
            inner: mat,
            _marker: PhantomData,
        }
    }

    pub fn apply_all<Z: TypeEncoder, B>(
        &self,
        unary_op: UnaryOp<T, Z>,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<T, T, T>>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z> {
        let (m, n) = self.shape();

        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let c = if let Some(Value::Transpose) = desc.get(Field::Input0) {
            SparseMatrix::<Z>::empty((n, m)) // no change at all
        } else {
            SparseMatrix::<Z>::empty((m, n)) // do the transpose
        };

        grb_run(|| unsafe {
            GrB_Matrix_apply(c.inner, mask, acc, unary_op.op, self.inner, desc.desc)
        });
        c
    }

    pub fn apply_mut<Z: TypeEncoder, B: CanBool>(
        self,
        unary_op: UnaryOp<T, Z>,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<T, T, T>>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z> {
        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        grb_run(|| unsafe {
            GrB_Matrix_apply(self.inner, mask, acc, unary_op.op, self.inner, desc.desc)
        });
        unsafe { mem::transmute(self) }
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

    pub fn transpose(&self) -> SparseMatrix<T> {
        self.transpose_all(empty_matrix_mask::<bool>(), None, &Descriptor::default())
    }

    pub fn transpose_all<B: CanBool>(
        &self,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<T, T, T>>,
        desc: &Descriptor,
    ) -> SparseMatrix<T> {
        let (m, n) = self.shape();

        let c = if let Some(Value::Transpose) = desc.get(Field::Input0) {
            SparseMatrix::<T>::empty((m, n)) // no change at all
        } else {
            SparseMatrix::<T>::empty((n, m)) // do the transpose
        };

        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        grb_run(|| unsafe { GrB_transpose(c.inner, mask, acc, self.inner, desc.desc) });
        c
    }

    pub fn extract(&self, l: Range<u64>, r: Range<u64>) -> SparseMatrix<T> {
        let is = vec![l.start, l.end - 1];
        let js = vec![r.start, r.end - 1];
        let c = SparseMatrix::<T>::empty((l.end, r.end));
        let d = Descriptor::default();
        grb_run(|| unsafe {
            GrB_Matrix_extract(
                c.inner,
                ptr::null_mut::<GB_Matrix_opaque>(),
                ptr::null_mut::<GB_BinaryOp_opaque>(),
                self.inner,
                is.as_ptr(),
                GxB_RANGE as u64,
                js.as_ptr(),
                GxB_RANGE as u64,
                d.desc,
            )
        });
        c
    }

    pub fn remove(&mut self, i: u64, j: u64) {
        grb_run(|| unsafe { GrB_Matrix_removeElement(self.inner, i, j) })
    }

    pub fn diag(
        &mut self,
        diag: &SparseVector<T>,
        k: i64,
        desc: Option<Descriptor>,
    ) -> &SparseMatrix<T> {
        let d = desc.unwrap_or(Descriptor::default());

        grb_run(|| unsafe { GxB_Matrix_diag(self.inner, diag.inner, k, d.desc) });

        self
    }
}

impl<T> SparseMatrix<T> {
    pub fn rows(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_nrows(G.as_mut_ptr(), self.inner) })
    }

    pub fn cols(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_ncols(G.as_mut_ptr(), self.inner) })
    }

    pub fn shape(&self) -> (u64, u64) {
        (self.rows(), self.cols())
    }

    pub fn nvals(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Matrix_nvals(G.as_mut_ptr(), self.inner) })
    }

    pub fn get_format(&self) -> MatrixFormat {
        let grb_fmt = grb_call(|F: &mut MaybeUninit<u32>| unsafe {
            GxB_Matrix_Option_get(self.inner, GxB_Option_Field_GxB_FORMAT, F.as_mut_ptr())
        });
        MatrixFormat::from_u32(grb_fmt).unwrap()
    }

    pub fn resize(&mut self, new_row: u64, new_col: u64) {
        grb_run(|| unsafe { GxB_Matrix_resize(self.inner, new_row, new_col) })
    }

    pub fn clear(&mut self) {
        grb_run(|| unsafe { GrB_Matrix_clear(self.inner) })
    }

    pub fn wait(&mut self) {
        grb_run(|| unsafe {
            GrB_Matrix_wait(self.inner, crate::ops::ffi::GrB_WaitMode_GrB_COMPLETE)
        })
    }
}

impl<T> Clone for SparseMatrix<T> {
    fn clone(&self) -> Self {
        let c = grb_call(|M: &mut MaybeUninit<GrB_Matrix>| unsafe {
            GrB_Matrix_dup(M.as_mut_ptr(), self.inner)
        });

        SparseMatrix {
            inner: c,
            _marker: PhantomData,
        }
    }
}

impl<T: TypeEncoder> SparseVector<T> {
    pub fn empty(size: u64) -> SparseVector<T> {
        let _ = *GRB;

        let vec = grb_call(|V: &mut MaybeUninit<GrB_Vector>| unsafe {
            GrB_Vector_new(V.as_mut_ptr(), T::blas_type().tpe, size)
        });
        SparseVector {
            inner: vec,
            _marker: PhantomData,
        }
    }
}

impl<T> SparseVector<T> {
    pub fn nvals(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Vector_nvals(G.as_mut_ptr(), self.inner) })
    }

    pub fn size(&self) -> u64 {
        grb_call(|G: &mut MaybeUninit<u64>| unsafe { GrB_Vector_size(G.as_mut_ptr(), self.inner) })
    }

    pub fn resize(&mut self, new_size: u64) {
        grb_run(|| unsafe { GxB_Vector_resize(self.inner, new_size) })
    }

    pub fn wait(&mut self) {
        grb_run(|| unsafe {
            GrB_Vector_wait(self.inner, crate::ops::ffi::GrB_WaitMode_GrB_COMPLETE)
        })
    }

    pub fn remove(&mut self, i: u64) {
        grb_run(|| unsafe { GrB_Vector_removeElement(self.inner, i) })
    }
}

impl<T> Clone for SparseVector<T> {
    fn clone(&self) -> Self {
        let c = grb_call(|M: &mut MaybeUninit<GrB_Vector>| unsafe {
            GrB_Vector_dup(M.as_mut_ptr(), self.inner)
        });

        SparseVector {
            inner: c,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.inner as *mut GrB_Matrix;
        self.wait();
        grb_run(|| unsafe { GrB_Matrix_free(m_pointer) })
    }
}

impl<T> Drop for SparseVector<T> {
    fn drop(&mut self) {
        let m_pointer = &mut self.inner as *mut GrB_Vector;
        self.wait();
        grb_run(|| unsafe { GrB_Vector_free(m_pointer) });
    }
}

macro_rules! partial_eq_impls {
    ( $typ:ty ) => {
        impl PartialEq for SparseMatrix<$typ> {
            fn eq(&self, other: &Self) -> bool {
                let shape = self.shape();
                if shape != other.shape() {
                    return false;
                }
                let nvals = self.nvals();
                if nvals != other.nvals() {
                    false
                } else {
                    let c = self.intersect(other, &BinaryOp::<$typ, $typ, bool>::eq());
                    if c.nvals() != nvals {
                        false
                    } else {
                        let mut result = false;
                        let m =
                            SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), true);
                        *c.reduce_all(&mut result, &m, None, None)
                    }
                }
            }
        }
        impl PartialEq for SparseVector<$typ> {
            fn eq(&self, other: &Self) -> bool {
                let size = self.size();
                if size != other.size() {
                    return false;
                }
                let nvals = self.nvals();
                if nvals != other.nvals() {
                    false
                } else {
                    let c = self.intersect(other, &BinaryOp::<$typ, $typ, bool>::eq());
                    if c.nvals() != nvals {
                        false
                    } else {
                        let mut result = false;
                        let m =
                            SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), true);
                        *c.reduce_all(&mut result, &m, None, None)
                    }
                }
            }
        }
    };
}

trait_gen_fn0!(partial_eq_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bool_sparse_matrix() {
        let mut m = SparseMatrix::<bool>::empty((5, 5));
        assert!(m.rows() == 5);
        assert!(m.insert((0, 3), true) == ());
        assert!(m.insert((1, 3), true) == ());
        assert!(m.insert((2, 3), true) == ());
        assert!(m.insert((3, 3), true) == ());
        assert!(m.insert((4, 3), true) == ());
        assert!(m.nvals() == 5);
        assert!(m.get((1, 3)) == Some(true));
        assert!(m.get((0, 0)) == None);
        assert!(m.get((1, 3)) == Some(true));
        assert!(m.get((2, 3)) == Some(true));
        assert!(m.get((0, 3)) == Some(true));
    }

    #[test]
    fn create_u64_sparse_matrix() {
        let mut m = SparseMatrix::<u64>::empty((2, 5));
        assert!(m.insert((0, 0), 12 as u64) == ());
        assert!(m.get((0, 0)) == Some(12 as u64));
    }

    #[test]
    fn create_u64_sparse_vector() {
        let mut v = SparseVector::<u64>::empty(10 as u64);
        assert!(v.insert(0, 3) == ());
        assert!(v.get(0) == Some(3));
    }

    #[test]
    fn vmx_bool() {
        let mut v = SparseVector::<bool>::empty(10);
        v.load(&[true, true, true], &[0, 4, 2]);

        let mut A = SparseMatrix::<bool>::empty((10, 10));
        A.insert((0, 0), true);
        A.insert((1, 0), true);
        A.insert((0, 1), true);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let land = BinaryOp::<bool, bool, bool>::land();
        let semi = Semiring::new(&m, land);

        v.vxm_mut::<bool, bool, bool, Descriptor>(
            empty_vector_mask::<bool>(),
            None,
            &A,
            &semi,
            None,
        );
    }

    #[test]
    fn reduce_vector_and_all_true() {
        let mut v = SparseVector::<bool>::empty(2);
        v.insert(0, true);
        v.insert(1, true);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), true);

        assert_eq!(*v.reduce_all(&mut true, &m, None, None), true);
    }

    #[test]
    fn reduce_vector_and_some_true_some_false() {
        let mut v = SparseVector::<bool>::empty(2);
        v.insert(0, true);

        let mut v2 = SparseVector::<bool>::empty(2);
        v2.insert(0, false);

        let m = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::land(), false);

        assert_eq!(*v.reduce_all(&mut true, &m, None, None), true);
        assert_eq!(*v2.reduce_all(&mut true, &m, None, None), false);
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
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        let mut v = SparseVector::<bool>::empty(7);
        v.insert(0, true);

        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());
        v.vxm_mut::<bool, bool, bool, Descriptor>(
            empty_vector_mask::<bool>(),
            None,
            &m,
            &or_and_semi,
            None,
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
    fn resize_changes_the_shape_of_a_vector() {
        let mut v = SparseVector::<bool>::empty(5);

        let s = v.size();
        assert_eq!(s, 5);

        v.resize(10);

        let s = v.size();
        assert_eq!(s, 10);
    }

    #[test]
    fn resize_changes_the_shape_of_a_matrix() {
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
    fn transpose_flips_the_shape_of_the_matrix() {
        let v = SparseMatrix::<bool>::empty((5, 7));

        let (r, c) = v.shape();
        assert_eq!(r, 5);
        assert_eq!(c, 7);

        let u = v.transpose();

        let (r, c) = u.shape();
        assert_eq!(r, 7);
        assert_eq!(c, 5);
    }

    #[test]
    fn clone_a_matrix() {
        let mut m = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        m.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        let n = m.clone();

        assert_eq!(n.get((0, 1)), Some(true));
        assert_eq!(n.get((0, 3)), Some(true));
        assert_eq!(n.get((1, 6)), Some(true));
        assert_eq!(n.get((1, 4)), Some(true));
        assert_eq!(n.get((2, 5)), Some(true));
    }

    #[test]
    fn extract_sub_matrix() {
        let mut m = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        m.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        let mut n = m.extract(0..4, 0..4);
        assert_eq!(n.shape(), (4, 4));

        for i in 0..4 {
            for j in 0..4 {
                if i != 0 {
                    if j != 1 || j != 3 {
                        assert_eq!(n.get((i, j)), None);
                    }
                }
            }
        }

        assert_eq!(n.get((0, 1)), Some(true));
        assert_eq!(n.get((0, 3)), Some(true));

        n.insert((1, 1), true);
        assert_eq!(n.get((1, 1)), Some(true));
        assert_eq!(m.get((1, 1)), None); // extract does a copy, it is not a view
    }

    #[test]
    fn partial_eq_test_equality() {
        let mut a = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        a.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        let mut b = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        b.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        assert_eq!(a, b)
    }

    #[test]
    fn partial_eq_test_in_equality() {
        let mut a = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        a.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 1, 4, 5, 1, 2, 3],
        );

        let mut b = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        b.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        assert_ne!(a, b)
    }

    #[test]
    fn partial_eq_test_in_equality_empty_on_shape() {
        let a = SparseMatrix::<bool>::empty((7, 7));
        let b = SparseMatrix::<bool>::empty((7, 7));
        let c = SparseMatrix::<bool>::empty((2, 5));

        assert_eq!(a, b);
        assert_ne!(b, c)
    }

    #[test]
    fn partial_eq_vector_empty_equality() {
        let v1 = SparseVector::<bool>::empty(5);
        let v2 = SparseVector::<bool>::empty(5);

        assert_eq!(v2, v1);
    }

    #[test]
    fn partial_eq_vector_empty_in_equality() {
        let v1 = SparseVector::<bool>::empty(4);
        let v2 = SparseVector::<bool>::empty(5);

        assert_ne!(v2, v1);
    }

    #[test]
    fn partial_eq_vector_equality() {
        let mut v1 = SparseVector::<bool>::empty(5);
        let mut v2 = SparseVector::<bool>::empty(5);

        v1.insert(3, true);
        v1.insert(0, true);
        v2.insert(3, true);
        v2.insert(0, true);

        assert_eq!(v2, v1);
    }

    #[test]
    fn partial_eq_vector_in_equality() {
        let mut v1 = SparseVector::<bool>::empty(5);
        let mut v2 = SparseVector::<bool>::empty(5);

        v1.insert(3, true);
        v1.insert(0, true);
        v2.insert(3, true);
        v2.insert(1, true);

        assert_ne!(v2, v1);
    }

    #[test]
    fn expand_from_one_node_to_next_node_in_single_edge_graph() {
        let mut edges = SparseMatrix::<bool>::empty((2, 2));
        edges.insert((0, 1), true);

        //start at 0
        let mut front = SparseVector::<bool>::empty(2);
        front.insert(0, true);

        //expand
        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());

        let next_front = front.vxm_mut::<bool, bool, bool, Descriptor>(
            empty_vector_mask::<bool>(),
            None,
            &edges,
            &or_and_semi,
            None,
        );

        assert_eq!(next_front.get(1), Some(true));
        assert_eq!(next_front.get(0), None);
    }

    #[test]
    fn expand_from_one_node_to_next_two_edge_graph() {
        // P1 --friend--> P0 --pet--> D2
        let mut pet = SparseMatrix::<bool>::empty((3, 3));
        pet.insert((0, 2), true);

        let mut friend = SparseMatrix::<bool>::empty((3, 3));
        friend.insert((1, 0), true);

        //start at 0 and find nodes with pets
        let mut have_pets = SparseMatrix::<bool>::empty((3, 3));
        have_pets.insert((0, 0), true);
        have_pets.insert((1, 1), true);

        //expand
        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());
        let or_and_semi2 = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());

        let next_have_pets = have_pets.mxm(
            empty_matrix_mask::<bool>(),
            None,
            &pet,
            or_and_semi,
            &Descriptor::default(),
        );

        assert_eq!(next_have_pets.get((0, 0)), None);
        assert_eq!(next_have_pets.get((0, 1)), None);
        assert_eq!(next_have_pets.get((0, 2)), Some(true));
        assert_eq!(next_have_pets.get((1, 0)), None);
        assert_eq!(next_have_pets.get((1, 1)), None);
        assert_eq!(next_have_pets.get((1, 2)), None);

        let mut dog_node_mat = SparseMatrix::<bool>::empty((3, 3));
        dog_node_mat.insert((2, 2), true);

        let extract_pets = next_have_pets.mxm(
            empty_matrix_mask::<bool>(),
            None,
            &dog_node_mat,
            or_and_semi2,
            &Descriptor::default(),
        );

        assert_eq!(extract_pets.get((0, 2)), Some(true));
    }

    #[test]
    fn test_vector_extract() {
        let mut v1 = SparseVector::<bool>::empty(5);

        v1.insert(3, true);
        v1.insert(0, true);

        let (vals, is) = v1.extract_tuples();

        assert_eq!(is, vec!(0, 3));
        assert_eq!(vals, vec!(true, true));
    }

    #[test]
    fn test_matrix_apply() {
        let mut mat1 = SparseMatrix::<u32>::empty((3, 5));
        mat1.insert((2, 4), 12);

        let mat2 = mat1.apply_all::<u32, bool>(
            UnaryOp::<u32, u32>::one(),
            None,
            None,
            &Descriptor::default(),
        );

        assert_eq!(mat2.shape(), (3, 5));
        assert_eq!(mat2.get((2, 4)), Some(1));
    }
    #[test]
    fn test_matrix_apply_transpose() {
        let mut mat1 = SparseMatrix::<u32>::empty((3, 5));
        mat1.insert((2, 4), 12);

        let mat2 = mat1.apply_all::<u32, bool>(
            UnaryOp::<u32, u32>::one(),
            None,
            None,
            &Descriptor::default().set(Field::Input0, Value::Transpose),
        );

        assert_eq!(mat2.shape(), (5, 3));
        assert_eq!(mat2.get((4, 2)), Some(1));
    }

    #[test]
    fn test_matrix_apply_transpose_on_self() {
        let mut mat1 = SparseMatrix::<u32>::empty((3, 5));
        mat1.insert((2, 4), 12);

        let mat2 = mat1.apply_mut::<u32, bool>(
            UnaryOp::<u32, u32>::one(),
            None,
            None,
            &Descriptor::default(),
        );

        assert_eq!(mat2.shape(), (3, 5));
        assert_eq!(mat2.get((2, 4)), Some(1));
    }

    #[test]
    fn test_matrix_apply_transpose_on_self_fail_java() {
        let mut mat1 = SparseMatrix::<i32>::empty((1, 3));
        mat1.insert((0, 0), -3);
        mat1.insert((0, 2), -4);
        mat1.insert((0, 1), 10);

        mat1.wait();

        let mut mat2 = mat1.apply_mut::<i32, bool>(
            UnaryOp::<i32, i32>::one(),
            None,
            None,
            &Descriptor::default(),
        );

        mat2.wait();

        assert_eq!(mat2.shape(), (1, 3));
        assert_eq!(mat2.get((0, 0)), Some(1));
        assert_eq!(mat2.get((0, 2)), Some(1));
        assert_eq!(mat2.get((0, 1)), Some(1));
    }

    #[test]
    fn test_matrix_apply_different_output() {
        let mut mat1 = SparseMatrix::<i32>::empty((1, 3));
        mat1.insert((0, 0), -3);
        mat1.insert((0, 2), -4);
        mat1.insert((0, 1), 10);

        mat1.wait();

        let mut mat2 = mat1.apply_all::<i32, bool>(
            UnaryOp::<i32, i32>::one(),
            None,
            None,
            &Descriptor::default(),
        );

        mat2.wait();

        assert_eq!(mat2.shape(), (1, 3));
        assert_eq!(mat2.get((0, 0)), Some(1));
        assert_eq!(mat2.get((0, 2)), Some(1));
        assert_eq!(mat2.get((0, 1)), Some(1));
    }

    #[test]
    fn test_update_matrix_in_parallel() {
        let n = 10;
        let sync_mat = SyncSparseMatrix::from_mat(SparseMatrix::<bool>::new(
            (n, n),
            &[true, true, true],
            &[0, 3, 9],
            &[1, 4, 8],
        ));
        for i in 0..n {
            let mut m = sync_mat.clone();

            std::thread::spawn(move || {
                m.use_mut(|mat| {
                    mat.insert((i, i), true);
                    let monoid =
                        SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), true);
                    let mut out = true;
                    mat.reduce_all(&mut out, &monoid, None, None);
                    assert_eq!(out, true);
                });
            });
        }
    }
}
