#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::ops::ffi::*;
use std::marker::PhantomData;

pub struct BinaryOp<A, B, C> {
    pub op: GrB_BinaryOp,
    _a: PhantomData<*const A>,
    _b: PhantomData<*const B>,
    _c: PhantomData<*const C>,
}

pub trait Lor<T> {
    fn lor() -> BinaryOp<T, T, T>;
}

impl Lor<bool> for BinaryOp<bool, bool, bool> {
    fn lor() -> BinaryOp<bool, bool, bool> {
        unsafe {
            BinaryOp {
                op: GxB_LOR_BOOL,
                _a: PhantomData,
                _b: PhantomData,
                _c: PhantomData,
            }
        }
    }
}

pub trait Land<T> {
    fn land() -> BinaryOp<T, T, T>;
}

impl Land<bool> for BinaryOp<bool, bool, bool> {
    fn land() -> BinaryOp<bool, bool, bool> {
        unsafe {
            BinaryOp {
                op: GxB_LAND_BOOL,
                _a: PhantomData,
                _b: PhantomData,
                _c: PhantomData,
            }
        }
    }
}
