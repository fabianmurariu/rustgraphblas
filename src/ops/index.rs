use crate::*;

pub trait Get<Idx> {
    type Item;
    fn get(&self, idx: Idx) -> Option<Self::Item>;
}

pub trait Insert<Idx> {
    type Item;
    fn insert(&mut self, idx: Idx, val: Self::Item);
}

pub trait Extract {
    type Item;
    type Indices;
    fn extract_tuples(&self) -> (Vec<Self::Item>, Self::Indices);
}

macro_rules! matrix_extract_impls {
    ( $typ:ty, $extract_elem_func:ident ) => {
        impl Extract for SparseMatrix<$typ> {
            type Item = $typ;
            type Indices = (Vec<u64>, Vec<u64>);

            fn extract_tuples(&self) -> (Vec<Self::Item>, Self::Indices) {
                let size = self.nvals();
                let mut is = Vec::with_capacity(size as usize);
                let mut js = Vec::with_capacity(size as usize);
                let mut vs = Vec::with_capacity(size as usize);

                let is_ptr = is.as_mut_ptr();
                let js_ptr = js.as_mut_ptr();
                let v_ptr = vs.as_mut_ptr();
                let mut nvals = self.nvals();

                grb_run(|| unsafe {
                    let grb_code =
                        $extract_elem_func(is_ptr, js_ptr, v_ptr, &mut nvals, self.inner);
                    is.set_len(nvals as usize);
                    js.set_len(nvals as usize);
                    vs.set_len(nvals as usize);
                    grb_code
                });

                (vs, (is, js))
            }
        }
    };
}

macro_rules! vector_extract_impls {
    ( $typ:ty, $extract_elem_func:ident ) => {
        impl Extract for SparseVector<$typ> {
            type Item = $typ;
            type Indices = Vec<u64>;
            fn extract_tuples(&self) -> (Vec<$typ>, Vec<u64>) {
                let size = self.nvals();
                let mut is = Vec::with_capacity(size as usize);
                let mut vs = Vec::with_capacity(size as usize);

                let is_ptr = is.as_mut_ptr();
                let v_ptr = vs.as_mut_ptr();
                let mut nvals = self.nvals();

                grb_run(|| unsafe {
                    let grb_code = $extract_elem_func(is_ptr, v_ptr, &mut nvals, self.inner);
                    is.set_len(nvals as usize);
                    vs.set_len(nvals as usize);
                    grb_code
                });

                (vs, is)
            }
        }
    };
}

macro_rules! matrix_insert_impls {
    ( $typ:ty, $set_elem_func:ident ) => {
        impl Insert<(u64, u64)> for SparseMatrix<$typ> {
            type Item = $typ;

            fn insert(&mut self, idx: (u64, u64), val: Self::Item) {
                let (row, col) = idx;
                grb_run(|| unsafe { $set_elem_func(self.inner, val, row, col) })
            }
        }
    };
}

macro_rules! vector_insert_impls {
    ( $typ:ty, $set_elem_func:ident ) => {
        impl Insert<u64> for SparseVector<$typ> {
            type Item = $typ;

            fn insert(&mut self, idx: u64, val: Self::Item) {
                grb_run(|| unsafe { $set_elem_func(self.inner, val, idx) })
            }
        }
    };
}

macro_rules! matrix_index_impls {
    ( $typ:ty, $get_elem_func:ident ) => {
        impl Get<(u64, u64)> for SparseMatrix<$typ> {
            type Item = $typ;

            fn get(&self, idx: (u64, u64)) -> Option<Self::Item> {
                let (i, j) = idx;
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.inner, i, j) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}, {}) GrB_error: {}", i, j, e),
                    }
                }
            }
        }
    };
}

macro_rules! vector_index_impls {
    ( $typ:ty, $get_elem_func:ident ) => {
        impl Get<u64> for SparseVector<$typ> {
            type Item = $typ;

            fn get(&self, idx: u64) -> Option<Self::Item> {
                let mut P = MaybeUninit::<$typ>::uninit();
                unsafe {
                    match $get_elem_func(P.as_mut_ptr(), self.inner, idx) {
                        0 => Some(P.assume_init()),
                        1 => None,
                        e => panic!("Failed to get element at ({}) GrB_error: {}", idx, e),
                    }
                }
            }
        }
    };
}

trait_gen_fn1!(extractElement; GrB_Matrix; matrix_index_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(extractElement; GrB_Vector; vector_index_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(setElement; GrB_Matrix; matrix_insert_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(setElement; GrB_Vector; vector_insert_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(extractTuples; GrB_Matrix; matrix_extract_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(extractTuples; GrB_Vector; vector_extract_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);
