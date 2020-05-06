# rustgraphblas

Wrapper for `GraphBLAS.h` exposing a nicer rust API

Exposes a set of routines over sparse matrices and sparse vectors combined with
various semirings. This allows graphs to be represented as sparse matrices and
various algorithms (bfs, connected components, page rank, ..) to be implemented
as a set of linear algebra operations.

More about GraphBLAS [here](http://graphblas.org/index.php?title=Graph_BLAS_Forum) 

Requirements: build and install GraphBLAS dependency, for details see[here](https://github.com/DrTimothyAldenDavis/GraphBLAS/blob/stable/README.md "GraphBLAS readme") 

```bash
cd deps/GraphBLAS
make clean install
```

Example of BFS from [bfs5m.c](https://github.com/fabianmurariu/SuiteSparse/blob/master/GraphBLAS/Demo/Source/bfs5m.c#L33)
```rust
/**
 * this is the test for the graph on the cover of 
 * Graph Algorithms in the Language of Linear Algebra
 * where by multiplying a boolean matrix with 
 * a boolean vector on the and/or semiring until there are no successor we get BFS
 * */
fn graph_blas_port_bfs(){
    let s:u64 = 0; // start at 0
    let n = 7; //vertices

    let mut A = SparseMatrix::<bool>::empty((n, n));

    let edges_n:usize = 10;
    A.load(edges_n as u64, &vec![true; edges_n],
           &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
           &[1, 3, 6, 4, 5, 2, 5, 2, 2, 3]);

    let mut v = SparseVector::<i32>::empty(n);
    let mut q = SparseVector::<bool>::empty(n);

    let mut default_desc = Descriptor::default();

    // GrB_assign (v, NULL, NULL, 0, GrB_ALL, n, NULL) ;   // make v dense
    v.assign_all(empty_mask::<bool>(), None, 0, n, &default_desc);

    //finish pending work on v
    assert_eq!(n, v.nvals());
    // GrB_Vector_setElement (q, true, s) ;   // q[s] = true, false elsewhere
    q.insert(s, true);

    // GrB_Monoid_new (&Lor, GrB_LOR, (bool) false) ;
    // GrB_Semiring_new (&Boolean, Lor, GrB_LAND) ;
    // FIXME: Semirings do not OWN monoids
    let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
    let lor_monoid2 = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
    let or_and_semi = Semiring::new(lor_monoid, BinaryOp::<bool, bool, bool>::land());


    let mut desc = Descriptor::default();
    desc.set(Field::Mask, Value::SCMP).set(Field::Output, Value::Replace);

    let mut successor = true;

    let mut level:i32 = 1;
    while successor && level <= (n as i32) {
        v.assign_all(Some(&q), None, level, n, &default_desc);

        q.vxm(Some(&v), None, &A, &or_and_semi, &desc);

        q.reduce(&mut successor, None, &lor_monoid2, &default_desc);

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

```
