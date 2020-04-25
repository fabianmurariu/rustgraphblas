use crate::*;

use std::ptr;

pub fn empty_matrix_mask<B>() -> Option<&'static SparseMatrix<B>> {
    None::<&SparseMatrix<B>>
}

pub trait MatrixXMatrix<X> {
    // A

    fn mxm<Y, Z: TypeEncoder, B: CanBool>(
        &self,
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>, // B
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z>; // C
}

impl<X: TypeEncoder> MatrixXMatrix<X> for SparseMatrix<X> {
    fn mxm<Y, Z: TypeEncoder, B: CanBool>(
        &self,                          // A
        mask: Option<&SparseMatrix<B>>, // any type that can be made boolean
        accum: Option<&BinaryOp<Z, Z, Z>>,
        B: &SparseMatrix<Y>, // B
        s_ring: Semiring<X, Y, Z>,
        desc: &Descriptor,
    ) -> SparseMatrix<Z> // C
    {
        let (m, _) = self.shape();
        let (_, n) = B.shape();

        let mask = mask
            .map(|x| x.inner)
            .unwrap_or(ptr::null_mut::<GB_Matrix_opaque>());
        let acc = accum
            .map(|x| x.op)
            .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

        let C = SparseMatrix::<Z>::empty((m, n)); // this is actually mutated by the row below
        grb_run(|| unsafe { GrB_mxm(C.inner, mask, acc, s_ring.s, self.inner, B.inner, desc.desc) });
        C
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
   
    #[test]
    fn multiply_2_matrices_with_mxm_for_bfs_no_transpose() {
        let mut A = SparseMatrix::<bool>::empty((7, 7));

        let edges_n: usize = 10;
        A.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3],
        );

        // get the neighbours for 0 and 6
        let mut B = SparseMatrix::<bool>::empty((2, 7));
        B.insert((0, 0), true);
        B.insert((1, 6), true);

        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());

        let C = B.mxm(
            empty_matrix_mask::<bool>(),
            None,
            &A,
            or_and_semi,
            &Descriptor::default(),
        );
        let (r, c) = C.shape();

        // the shape must match 10x2
        assert_eq!(r, 2);
        assert_eq!(c, 7);

        let n = vec![(0, 1), (0, 3), (1, 2), (1, 3)];
        let neighbours = n.iter().fold(HashSet::new(), |mut set, x| {
            set.insert(x);
            set
        });

        for i in 0..r {
            for j in 0..c {
                let x = C.get((i, j));
                if neighbours.contains(&(i, j)) {
                    assert_eq!(x, Some(true));
                } else {
                    assert_eq!(x, None);
                }
            }
        }
    }

    #[test]
    fn graph_blas_port_bfs() {
        let s: u64 = 0; // start at 0
        let n = 7; //vertices

        let mut a = SparseMatrix::<bool>::empty((n, n));

        let edges_n: usize = 10;
        a.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 2, 5, 2, 2, 3],
        );

        let mut v = SparseVector::<i32>::empty(n);
        let mut q = SparseVector::<bool>::empty(n);

        let default_desc = Descriptor::default();

        // GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL) ;   // make v dense
        v.assign_all(empty_vector_mask::<bool>(), None, 0, n, &default_desc);

        //finish pending work on v
        assert_eq!(n, v.nvals());
        // GrB_Vector_setElement (q, true, s) ;   // q[s] = true, false elsewhere
        q.insert(s, true);

        // GrB_Monoid_new (&Lor, GrB_LOR, (bool) false) ;
        // GrB_Semiring_new (&Boolean, Lor, GrB_LAND) ;
        let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
        let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());

        let mut desc = Descriptor::default();
        desc.set(Field::Mask, Value::SCMP)
            .set(Field::Output, Value::Replace);

        let mut successor = true;

        let mut level: i32 = 1;
        while successor && level <= (n as i32) {

            v.assign_all(Some(&q), None, level, n, &default_desc);

            q.vxm_mut(Some(&v), None, &a, &or_and_semi, Some(&desc));

            q.reduce_all(&mut successor, &lor_monoid, None, None);

            level = level + 1;
        }

        assert_eq!(v.get(0), Some(1));

        assert_eq!(v.get(1), Some(2));
        assert_eq!(v.get(3), Some(2));

        assert_eq!(v.get(4), Some(3));
        assert_eq!(v.get(6), Some(3));
        assert_eq!(v.get(2), Some(3));

        assert_eq!(v.get(5), Some(4));
    }

    #[test]
    fn graph_blas_port_strong_cc(){

        // like bfs but not tracking the distance just marks the reachable nodes
        fn simple_bfs(g:&SparseMatrix<bool>, v:u64) -> SparseVector<bool>{
            let (size, _) = g.shape();
            let mut frontier = SparseVector::<bool>::empty(size);
            let mut reached = SparseVector::<bool>::empty(size);
            frontier.insert(v, true);

            let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
            let or_and_semi = Semiring::new(&lor_monoid, BinaryOp::<bool, bool, bool>::land());

            let default_desc = Descriptor::default();

            let mut desc = Descriptor::default();
            desc.set(Field::Mask, Value::SCMP)
                .set(Field::Output, Value::Replace);

            loop {
                reached.assign_all(Some(&frontier), None, true, size, &default_desc); // set all of reached to x
                frontier.vxm_mut(Some(&reached), None, &g, &or_and_semi, Some(&desc));
                if frontier.nvals() == 0 {
                    return reached //return all nodes reached
                }
            }
        }

        let n = 7;
        let mut a = SparseMatrix::<bool>::empty((n, n));

        let edges_n: usize = 10;
        a.load(
            &vec![true; edges_n],
            &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
            &[1, 3, 6, 4, 5, 2, 5, 2, 2, 3],
        );

        let pred = simple_bfs(&a, 1);

        assert_eq!(pred.get(1), Some(true));
        assert_eq!(pred.get(3), Some(true));
    }
}
