use crate::*;

pub trait Reduce<T> {

    fn reduce_all<'m>(&self, item:&'m mut T, monoid:&SparseMonoid<T>, accum:Option<&BinaryOp<T, T, T>>, desc:Option<&Descriptor>) -> &'m T;
}

macro_rules! matrix_reduce_impls {
    ( $typ:ty, $grb_reduce_fun:ident ) => {

        impl Reduce<$typ> for SparseMatrix<$typ> {

            fn reduce_all<'m>(&self, item:&'m mut $typ, monoid:&SparseMonoid<$typ>, accum:Option<&BinaryOp<$typ, $typ, $typ>>, desc:Option<&Descriptor>) -> &'m $typ {
                unsafe {
                    let acc = accum
                        .map(|x| x.op)
                        .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

                    grb_run(|| {
                        $grb_reduce_fun(item, acc, monoid.m, self.inner, desc.map(|d|d.desc).unwrap_or(Descriptor::default().desc))
                    });

                    item
                }
            }
        }
    }
}

macro_rules! vector_reduce_impls {
    ( $typ:ty, $grb_reduce_fun:ident ) => {

        impl Reduce<$typ> for SparseVector<$typ> {

            fn reduce_all<'m>(&self, item:&'m mut $typ, monoid:&SparseMonoid<$typ>, accum:Option<&BinaryOp<$typ, $typ, $typ>>, desc:Option<&Descriptor>) -> &'m $typ {
                unsafe {
                    let acc = accum
                        .map(|x| x.op)
                        .unwrap_or(ptr::null_mut::<GB_BinaryOp_opaque>());

                    grb_run(|| {
                        $grb_reduce_fun(item, acc, monoid.m, self.inner, desc.map(|d|d.desc).unwrap_or(Descriptor::default().desc))
                    });

                    item
                }
            }
        }
    }
}

trait_gen_fn1!(reduce; GrB_Matrix; matrix_reduce_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);

trait_gen_fn1!(reduce; GrB_Vector; vector_reduce_impls;
    bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64;
    BOOL, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP32, FP64);
