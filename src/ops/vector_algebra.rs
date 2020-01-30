use crate::ops::ffi::*;
use crate::{SparseVector, SparseMatrix, VectorLike};
use crate::ops::types::*;
use crate::ops::types::desc::*;
use crate::ops::binops::*;
use crate::ops::monoid::*;
use std::ptr;

pub fn empty_mask<B>() -> Option<&'static SparseVector<B>> {
    None::<&SparseVector<B>>
}

pub trait VectorAlgebra<Z> {

    fn vxm<X, Y, B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        m: &SparseMatrix<Y>,
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;

}

pub trait Assign<Z> {

    fn assign<B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        x: Z, 
        indices: &[u64],
        ni: u64,
        desc: Descriptor
    ) -> &SparseVector<Z>;
}

macro_rules! make_vector_assign {
    ( $rust_typ:ty, $grb_assign_fn:ident ) => {
       impl Assign<$rust_typ> for SparseVector<$rust_typ> {

        fn assign<B:CanBool>(
            &mut self,
            mask: Option<&SparseVector<B>>, // any type that can be made boolean
            accum: Option<&BinaryOp<$rust_typ, $rust_typ, $rust_typ>>,
            x: $rust_typ, 
            indices: &[u64],
            ni: u64,
            desc: Descriptor
        ) -> &SparseVector<$rust_typ> {
            
            let mask = mask.map(|x| x.vec).unwrap_or(ptr::null_mut::<GB_Vector_opaque>()); 
            let acc = accum.map(|x| x.op).unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());
            unsafe {
                $grb_assign_fn(self.vec, mask, acc, x, indices.as_ptr(), ni, desc.desc);                
            }
            self
        }

       } 
    };
}

macro_rules! make_vector_assign_all{
    ( $rust_maker:ident; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                $rust_maker!(
                    $rust_tpe,
                    [<GrB_Vector_assign_ $grb_tpe>]
                );
                )*
        }
    }
}

make_vector_assign_all!(make_vector_assign; bool, u64; BOOL, UINT64);

#[test]
fn create_sparse_vector_assign_subvector(){

    let mut v = SparseVector::<u64>::empty(10);
    v.insert(0, 32);
    v.insert(1, 12);

    assert_eq!(v.get(2), None);
    assert_eq!(v.get(3), None);
    assert_eq!(v.get(4), None);

    assert_eq!(v.nvals(), 2);

    v.assign(empty_mask::<bool>(), None, 11, &vec!(2, 3, 4), 3, Descriptor::default());

    assert_eq!(v.nvals(), 5);


    assert_eq!(v.get(2), Some(11));
    assert_eq!(v.get(3), Some(11));
    assert_eq!(v.get(4), Some(11));
}


impl <Z> VectorAlgebra<Z> for SparseVector<Z> {

    fn vxm<X, Y, B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        m: &SparseMatrix<Y>,
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor
    ) -> &SparseVector<Z> {

        let mask = mask.map(|x| x.vec).unwrap_or(ptr::null_mut::<GB_Vector_opaque>()); 
        let acc = accum.map(|x| x.op).unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());
        unsafe {
            match GrB_vxm(self.vec, mask, acc, s_ring.s, self.vec, m.mat, desc.desc){
                0 => self,
                err => panic!("VXM failed GrB_error {}", err)
            }
        }
    }

}


pub trait Reduce<T> {
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
            let m = ptr::null_mut::<GB_BinaryOp_opaque>();
            let op_acc = acc.map(|x| x.op).unwrap_or(m);
            GrB_Vector_reduce_BOOL(init, op_acc, monoid.m, self.vec, desc.desc);
        }
        init
    }
}
