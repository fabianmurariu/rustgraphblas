use crate::ops::binops::*;
use crate::ops::ffi::*;
use crate::ops::monoid::*;
use crate::ops::types::desc::*;
use crate::ops::types::*;
use crate::*;
use std::ptr;
use std::mem;
use either::*;

pub fn empty_mask<B>() -> Option<&'static SparseVector<B>> {
    None::<&SparseVector<B>>
}

#[macro_export]
macro_rules! grb_trait_constructor{
    ( $rust_maker:ident; $grb_partial_name:ident; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                $rust_maker!(
                    $rust_tpe,
                    [<$grb_partial_name $grb_tpe>]
                );
                )*
        }
    }
}

pub trait VectorBuilder<Z: TypeEncoder> {
    fn load(&mut self, n: u64, zs: &[Z], is: &[u64]) -> &SparseVector<Z>;
}

macro_rules! make_vector_builder {
    ( $rust_typ:ty, $grb_assign_fn:ident ) => {
        impl VectorBuilder<$rust_typ> for SparseVector<$rust_typ> {
            fn load(&mut self, n: u64, zs: &[$rust_typ], is: &[u64]) -> &SparseVector<$rust_typ> {
                grb_run(|| {
                    let dup = BinaryOp::<$rust_typ, $rust_typ, $rust_typ>::first();
                    unsafe { $grb_assign_fn(self.vec, is.as_ptr(), zs.as_ptr(), n, dup.op) }
                });
                self
            }
        }
    };
}

grb_trait_constructor!(make_vector_builder; GrB_Vector_build_;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

pub trait MatrixBuilder<Z> {
    fn load(&mut self, n: u64, zs: &[Z], is: &[u64], js: &[u64]) -> &SparseMatrix<Z>;
}

macro_rules! make_matrix_builder {
    ( $rust_typ:ty, $grb_assign_fn:ident ) => {
        impl MatrixBuilder<$rust_typ> for SparseMatrix<$rust_typ> {
            fn load(
                &mut self,
                n: u64,
                zs: &[$rust_typ],
                is: &[u64],
                js: &[u64],
            ) -> &SparseMatrix<$rust_typ> {
                grb_run(|| {
                    let dup = BinaryOp::<$rust_typ, $rust_typ, $rust_typ>::first();
                    unsafe {
                        $grb_assign_fn(self.mat, is.as_ptr(), js.as_ptr(), zs.as_ptr(), n, dup.op)
                    }
                });
                self
            }
        }
    };
}

grb_trait_constructor!(make_matrix_builder; GrB_Matrix_build_;
                       bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
                       BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

pub trait VectorAlgebra<Z> {
    fn vxm<X, Y, B: CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        m: &SparseMatrix<Y>,
        s_ring: &Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;
}

pub trait Assign<Z> {
    fn assign<B: CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        x: Z,
        indices: &[u64],
        ni: u64,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;

    fn assign_all<B: CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        x: Z,
        ni: u64,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;
}

macro_rules! make_vector_assign {
    ( $rust_typ:ty, $grb_assign_fn:ident ) => {
        impl Assign<$rust_typ> for SparseVector<$rust_typ> {
            fn assign<B: CanBool>(
                &mut self,
                mask: Option<&SparseVector<B>>, // any type that can be made boolean
                accum: Option<&BinaryOp<$rust_typ, $rust_typ, $rust_typ>>,
                x: $rust_typ,
                indices: &[u64],
                ni: u64,
                desc: &Descriptor,
            ) -> &SparseVector<$rust_typ> {
                let mask = mask
                    .map(|x| x.vec)
                    .unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
                let acc = accum
                    .map(|x| x.op)
                    .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());
                unsafe {
                    $grb_assign_fn(self.vec, mask, acc, x, indices.as_ptr(), ni, desc.desc);
                }
                self
            }

            fn assign_all<B: CanBool>(
                &mut self,
                mask: Option<&SparseVector<B>>, // any type that can be made boolean
                accum: Option<&BinaryOp<$rust_typ, $rust_typ, $rust_typ>>,
                x: $rust_typ,
                ni: u64,
                desc: &Descriptor,
            ) -> &SparseVector<$rust_typ> {
                let mask = mask
                    .map(|x| x.vec)
                    .unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
                let acc = accum
                    .map(|x| x.op)
                    .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());
                unsafe {
                    $grb_assign_fn(self.vec, mask, acc, x, GrB_ALL, ni, desc.desc);
                }
                self
            }
        }
    };
}

grb_trait_constructor!(make_vector_assign; GrB_Vector_assign_;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

#[test]
fn create_sparse_vector_assign_subvector() {
    let mut v = SparseVector::<u64>::empty(10);
    v.insert(0, 32);
    v.insert(1, 12);

    assert_eq!(v.get(2), None);
    assert_eq!(v.get(3), None);
    assert_eq!(v.get(4), None);

    assert_eq!(v.nvals(), 2);

    v.assign(
        empty_mask::<bool>(),
        None,
        11,
        &vec![2, 3, 4],
        3,
        &Descriptor::default(),
    );

    assert_eq!(v.nvals(), 5);

    assert_eq!(v.get(2), Some(11));
    assert_eq!(v.get(3), Some(11));
    assert_eq!(v.get(4), Some(11));
}

#[test]
fn create_sparse_vector_make_dense_assign_all() {

    let mut v = SparseVector::<u64>::empty(10);
    v.insert(0, 32);
    v.insert(1, 12);

    assert_eq!(v.get(2), None);
    assert_eq!(v.get(3), None);
    assert_eq!(v.get(4), None);

    assert_eq!(v.nvals(), 2);

    v.assign_all(
        empty_mask::<bool>(),
        None,
        12,
        10,
        &Descriptor::default()
    );

    assert_eq!(v.nvals(), 10);

    // vector is now dense
    for i in 0..10 {
        assert_eq!(v.get(i), Some(12));
    }
}

impl<Z> VectorAlgebra<Z> for SparseVector<Z> {
    fn vxm<X, Y, B: CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        m: &SparseMatrix<Y>,
        s_ring: &Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> &SparseVector<Z> {
        let mask = mask
            .map(|x| x.vec)
            .unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());
        grb_run(|| unsafe { GrB_vxm(self.vec, mask, acc, s_ring.s, self.vec, m.mat, desc.desc) });
        self
    }
}

pub trait Reduce<T> {
    fn reduce<'m>(
        &self,
        init: &'m mut T,
        acc: Option<BinaryOp<T, T, T>>,
        monoid: &SparseMonoid<T>,
        desc: &Descriptor,
    ) -> &'m T;
}

macro_rules! make_vector_reduce {
    ( $rust_typ:ty, $grb_reduce_fn:ident ) => {
        impl Reduce<$rust_typ> for SparseVector<$rust_typ> {
            fn reduce<'m>(
                &self,
                init: &'m mut $rust_typ,
                acc: Option<BinaryOp<$rust_typ, $rust_typ, $rust_typ>>,
                monoid: &SparseMonoid<$rust_typ>,
                desc: &Descriptor,
            ) -> &'m $rust_typ {
                unsafe {
                    let m = ptr::null_mut::<GB_BinaryOp_opaque>();
                    let op_acc = acc.map(|x| x.op).unwrap_or(m);
                    $grb_reduce_fn(init, op_acc, monoid.m, self.vec, desc.desc);
                }
                init
            }
        }
    };
}

grb_trait_constructor!(make_vector_reduce; GrB_Vector_reduce_;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

pub trait ElemWiseAlgebra<X> {
    fn mut_elem_wise_mult<Y, Z:TypeEncoder, B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor
    ) -> &SparseVector<Z>;

    fn elem_wise_mult<Y, Z:TypeEncoder, B:CanBool>(
        &self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor
    ) -> SparseVector<Z>;

    fn elem_wise_add<Y, Z:TypeEncoder, B:CanBool>(
        &self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor
    ) -> SparseVector<Z>;
}


impl <X:TypeEncoder> ElemWiseAlgebra<X> for SparseVector<X> {

    fn mut_elem_wise_mult<Y, Z:TypeEncoder, B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor
    ) -> &SparseVector<Z> {


        let mask = mask.map(|x| x.vec).unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum.map(|x| x.op).unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        grb_run(||{
            unsafe{
                match s_ring{
                    Left(semi) => GrB_eWiseMult_Vector_Semiring(self.vec, mask, acc, semi.s, self.vec, B.vec, desc.desc),
                    Right(op) => GrB_eWiseMult_Vector_BinaryOp(self.vec, mask, acc, op.op, self.vec, B.vec, desc.desc)
                }
            }
        });
        unsafe {
            mem::transmute(self)
        }
    }

    fn elem_wise_mult<Y, Z:TypeEncoder, B:CanBool>(
        &self, // A
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseVector<Z> // C
    {
        let s = self.size();

        let mask = mask.map(|x| x.vec).unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum.map(|x| x.op).unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let C = SparseVector::<Z>::empty(s); // this is actually mutated by the row below
        grb_run(||{
            unsafe{
                match s_ring{
                    Left(semi) => GrB_eWiseMult_Vector_Semiring(C.vec, mask, acc, semi.s, self.vec, B.vec, desc.desc),
                    Right(op) => GrB_eWiseMult_Vector_BinaryOp(C.vec, mask, acc, op.op, self.vec, B.vec, desc.desc)
                }
            }
        });
        C
    }

    fn elem_wise_add<Y, Z:TypeEncoder, B:CanBool>(
        &self, // A
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseVector<Z> // C
    {
        let s = self.size();

        let mask = mask.map(|x| x.vec).unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum.map(|x| x.op).unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let C = SparseVector::<Z>::empty(s); // this is actually mutated by the row below
        grb_run(||{
            unsafe{
                match s_ring {
                    Left(semi) => GrB_eWiseAdd_Vector_Semiring(C.vec, mask, acc, semi.s, self.vec, B.vec, desc.desc),
                    Right(op) => GrB_eWiseAdd_Vector_BinaryOp(C.vec, mask, acc, op.op, self.vec, B.vec, desc.desc),
                }
            }
        });
        C
    }
}

#[test]
fn elem_wise_add_i32() {
    let mut a = SparseVector::<i32>::empty(10);
    a.load(5, &[12, 34, -56, 78], &[0, 2, 4, 6, 8]);

    let mut b = SparseVector::<i32>::empty(10);
    b.load(5, &[8, -14, 36, -58], &[0, 2, 4, 6, 8]);

    let c = a.elem_wise_add(empty_mask::<bool>(), None, &b, Right(&BinaryOp::<i32, i32, i32>::plus()), &Descriptor::default());

    assert_eq!(c.get(0), Some(20));
    assert_eq!(c.get(1), None);
    assert_eq!(c.get(2), Some(20));
    assert_eq!(c.get(3), None);

   
}
