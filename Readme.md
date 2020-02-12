# rustgraphblas

Wrapper for `GraphBLAS.h` exposing a nicer rust API

Exposes a set of routines over sparse matrices and sparse vectors combined with
various semigroups. This allows graphs to be represented as sparse matrices and
various algorithms (bfs, connected components, page rank, ..) to be implemented
as a set of linear algebra operations.

More about GraphBLAS here[here](http://graphblas.org/index.php?title=Graph_BLAS_Forum) 

Example of 1 hop neighbours

```rust
/**
 * this is the test for the graph on the cover of 
 * Graph Algorithms in the Language of Linear Algebra
 * where by multiplying a boolean matrix with 
 * a boolean vector on the and/or semiring we effectively find the neighbours
 * for the input vertex
 * */
fn define_graph_adj_matrix_one_hop_neighbours(){
    let mut m = SparseMatrix::<bool>::empty((7, 7));

    let edges_n:usize = 10;
    m.load(edges_n as u64, &vec![true; edges_n],
           &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6],
           &[1, 3, 6, 4, 5, 4, 5, 4, 2, 3]);

    let mut v = SparseVector::<bool>::empty(7);
    v.insert(0, true);

    let lor_monoid = SparseMonoid::<bool>::new(BinaryOp::<bool, bool, bool>::lor(), false);
    let or_and_semi = Semiring::new(lor_monoid, BinaryOp::<bool, bool, bool>::land());
    v.vxm(empty_mask::<bool>(), None, &m, or_and_semi, &Descriptor::default());

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

```
