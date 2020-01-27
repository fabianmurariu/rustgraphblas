use crate::ops::ffi::*;
use crate::{SparseVector, SparseMatrix};
use crate::ops::types::*;
use crate::ops::types::desc::*;
use crate::ops::binops::*;
use crate::ops::monoid::*;
use std::ptr;

pub trait VectorAlgebra<Z> {

    fn vxm<X, Y, B:CanBool>(
        &mut self,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        m: &SparseMatrix<Y>,
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;

    // fn assign<B:CanBool>(
    //     &mut self,
    //     mask: Option<&SparseVector<B>>, // any type that can be made boolean
    //     accum: Option<&BinaryOp<Z, Z, Z>>,
    //     x: Z, 
    //     indices: &[u64],
    //     ni: u64,
    //     desc: Descriptor
    // ) -> &SparseVector<Z>;
}

pub fn empty_mask<B>() -> Option<&'static SparseVector<B>> {
    None::<&SparseVector<B>>
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