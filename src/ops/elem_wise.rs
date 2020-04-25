use crate::*;
use either::*;
use std::mem;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Items {
    Intersect,
    Union,
}

pub trait ElemWiseVector<X> {
    fn apply_mut<Y, Z: TypeEncoder, B: CanBool>(
        &mut self,
        items: Items,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> &SparseVector<Z>;

    fn apply<Y, Z: TypeEncoder, B: CanBool>(
        &self,
        items: Items,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseVector<Z>;

    fn union<Y, Z: TypeEncoder>(
        &self,
        b: &SparseVector<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> SparseVector<Z> {
        self.apply(
            Items::Union,
            empty_vector_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn intersect<Y, Z: TypeEncoder>(
        &self,
        b: &SparseVector<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> SparseVector<Z> {
        self.apply(
            Items::Intersect,
            empty_vector_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn union_mut<Y, Z: TypeEncoder>(
        &mut self,
        b: &SparseVector<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> &SparseVector<Z> {
        self.apply_mut(
            Items::Union,
            empty_vector_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn intersect_mut<Y, Z: TypeEncoder>(
        &mut self,
        b: &SparseVector<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> &SparseVector<Z> {
        self.apply_mut(
            Items::Intersect,
            empty_vector_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }
}

impl<X: TypeEncoder> ElemWiseVector<X> for SparseVector<X> {
    fn apply_mut<Y, Z: TypeEncoder, B: CanBool>(
        &mut self, // A
        items: Items,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> &SparseVector<Z> // C
    {
        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        grb_run(|| match items {
            Items::Union => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseAdd_Vector_Semiring(
                        self.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseAdd_Vector_BinaryOp(
                        self.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
            Items::Intersect => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseMult_Vector_Semiring(
                        self.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseMult_Vector_BinaryOp(
                        self.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
        });
        unsafe { mem::transmute(self) }
    }

    fn apply<Y, Z: TypeEncoder, B: CanBool>(
        &self, // A
        items: Items,
        mask: Option<&SparseVector<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseVector<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseVector<Z> // C
    {
        let s = self.size();

        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Vector_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let C = SparseVector::<Z>::empty(s); // this is actually mutated by the row below
        grb_run(|| match items {
            Items::Union => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseAdd_Vector_Semiring(
                        C.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseAdd_Vector_BinaryOp(
                        C.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
            Items::Intersect => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseMult_Vector_Semiring(
                        C.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseMult_Vector_BinaryOp(
                        C.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
        });
        C
    }
}

pub trait ElemWiseMatrix<X> {
    fn apply_mut<Y, Z: TypeEncoder, B: CanBool>(
        &mut self,
        items: Items,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> &SparseMatrix<Z>;

    fn apply<Y, Z: TypeEncoder, B: CanBool>(
        &self,
        items: Items,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>,
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z>;

    fn union<Y, Z: TypeEncoder>(
        &self,
        b: &SparseMatrix<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> SparseMatrix<Z> {
        self.apply(
            Items::Union,
            empty_matrix_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn intersect<Y, Z: TypeEncoder>(
        &self,
        b: &SparseMatrix<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> SparseMatrix<Z> {
        self.apply(
            Items::Intersect,
            empty_matrix_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn union_mut<Y, Z: TypeEncoder>(
        &mut self,
        b: &SparseMatrix<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> &SparseMatrix<Z> {
        self.apply_mut(
            Items::Union,
            empty_matrix_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }

    fn intersect_mut<Y, Z: TypeEncoder>(
        &mut self,
        b: &SparseMatrix<Y>,
        op: &BinaryOp<X, Y, Z>,
    ) -> &SparseMatrix<Z> {
        self.apply_mut(
            Items::Intersect,
            empty_matrix_mask::<bool>(),
            None,
            b,
            Right(op),
            &Descriptor::default(),
        )
    }
}

impl<X: TypeEncoder> ElemWiseMatrix<X> for SparseMatrix<X> {
    fn apply_mut<Y, Z: TypeEncoder, B: CanBool>(
        &mut self, // A
        items: Items,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> &SparseMatrix<Z> // C
    {
        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        grb_run(|| match items {
            Items::Union => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseAdd_Matrix_Semiring(
                        self.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseAdd_Matrix_BinaryOp(
                        self.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
            Items::Intersect => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseMult_Matrix_Semiring(
                        self.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseMult_Matrix_BinaryOp(
                        self.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
        });
        unsafe { mem::transmute(self) }
    }

    fn apply<Y, Z: TypeEncoder, B: CanBool>(
        &self, // A
        items: Items,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>, // B
        s_ring: Either<&Semiring<X, Y, Z>, &BinaryOp<X, Y, Z>>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z> // C
    {
        let (m, n) = self.shape();

        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let C = SparseMatrix::<Z>::empty((m, n)); // this is actually mutated by the row below
        grb_run(|| match items {
            Items::Union => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseAdd_Matrix_Semiring(
                        C.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseAdd_Matrix_BinaryOp(
                        C.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
            Items::Intersect => unsafe {
                match s_ring {
                    Left(semi) => GrB_eWiseMult_Matrix_Semiring(
                        C.inner, mask, acc, semi.s, self.inner, B.inner, desc.desc,
                    ),
                    Right(op) => GrB_eWiseMult_Matrix_BinaryOp(
                        C.inner, mask, acc, op.op, self.inner, B.inner, desc.desc,
                    ),
                }
            },
        });
        C
    }
}


#[cfg(test)]
mod tests {

    use super::*;
   
    #[test]
    fn element_wise_mult_A_or_B() {
        let mut a = SparseMatrix::<bool>::empty((2, 2));
        a.insert((0, 0), true);
        a.insert((1, 1), true);

        let mut b = SparseMatrix::<bool>::empty((2, 2));
        b.insert((1, 1), true);

        let or = BinaryOp::<bool, bool, bool>::lor();

        // mul is an intersection
        let c = a.apply(
            Items::Intersect,
            empty_matrix_mask::<bool>(),
            None,
            &b,
            Right(&or),
            &Descriptor::default(),
        );

        assert_eq!(c.get((0, 0)), None);
        assert_eq!(c.get((1, 1)), Some(true));
        assert_eq!(c.get((0, 1)), None);
        assert_eq!(c.get((1, 0)), None);
    }

    #[test]
    fn element_wise_add_A_or_B() {
        let mut a = SparseMatrix::<bool>::empty((2, 2));
        a.insert((0, 0), true);

        let mut b = SparseMatrix::<bool>::empty((2, 2));
        b.insert((1, 1), true);

        let or = BinaryOp::<bool, bool, bool>::lor();

        let c = a.apply(
            Items::Union,
            empty_matrix_mask::<bool>(),
            None,
            &b,
            Right(&or),
            &Descriptor::default(),
        );

        assert_eq!(c.get((0, 0)), Some(true));
        assert_eq!(c.get((1, 1)), Some(true));
        assert_eq!(c.get((0, 1)), None);
        assert_eq!(c.get((1, 0)), None);
    }

    #[test]
    fn element_wise_mut_eq_union_A_eq_B() {
        let mut a = SparseMatrix::<i32>::empty((3, 3));
        a.insert((0, 0), 3);
        a.insert((1, 1), 4);
        a.insert((2, 2), 5);

        let mut b = SparseMatrix::<i32>::empty((3, 3));
        b.insert((0, 0), 3);
        b.insert((1, 1), 4);
        b.insert((2, 2), -5);

        let bool_a = a.union_mut(&b, &BinaryOp::<i32, i32, bool>::eq());

        assert_eq!(bool_a.get((0, 0)), Some(true));
        assert_eq!(bool_a.get((1, 1)), Some(true));
        assert_eq!(bool_a.get((2, 2)), Some(false));
    }

    #[test]
    fn element_wise_add_A_and_B() {
        let mut a = SparseMatrix::<bool>::empty((2, 2));
        a.insert((0, 0), true);
        a.insert((0, 1), true);

        let mut b = SparseMatrix::<bool>::empty((2, 2));
        b.insert((1, 1), true);
        a.insert((0, 1), true);

        let and = BinaryOp::<bool, bool, bool>::land();

        // add is a union of a and b
        let c = a.apply(
            Items::Union,
            empty_matrix_mask::<bool>(),
            None,
            &b,
            Right(&and),
            &Descriptor::default(),
        );

        assert_eq!(c.get((0, 0)), Some(true)); // from a
        assert_eq!(c.get((1, 1)), Some(true)); // from b
        assert_eq!(c.get((0, 1)), Some(true)); // from a and b
        assert_eq!(c.get((1, 0)), None); // not present
    }

    #[test]
    fn elem_wise_add_i32() {
        let mut a = SparseVector::<i32>::empty(10);
        a.load(&[12, 34, -56, 78], &[0, 2, 4, 6]);

        let mut b = SparseVector::<i32>::empty(10);
        b.load(&[8, -14, 36, -58], &[0, 2, 4, 6]);

        let c = a.union_mut(
            &b,
            &BinaryOp::<i32, i32, i32>::plus(),
        );

        assert_eq!(c.get(0), Some(20));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(20));
        assert_eq!(c.get(3), None);
    }

    #[test]
    fn in_place_elem_wise_mul_changes_the_vector_type() {
        let mut a = SparseVector::<i32>::empty(10);
        a.load(&[1, 2, -3, -5], &[0, 2, 4, 8]);

        let mut b = SparseVector::<i32>::empty(10);
        b.load(&[1, 2, -3, -4], &[0, 2, 4, 8]);

        let c = a.intersect_mut(
            &b,
            &BinaryOp::<i32, i32, bool>::eq(),
        );

        assert_eq!(c.get(0), Some(true));
        assert_eq!(c.get(2), Some(true));
        assert_eq!(c.get(4), Some(true));
        assert_eq!(c.get(8), Some(false));
    }
}
