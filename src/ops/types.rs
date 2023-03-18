#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::{ops::ffi::*, Udf};
use std::mem::size_of;
use std::mem::MaybeUninit;

pub struct SparseType {
    pub(crate) tpe: GrB_Type,
}

pub trait TypeEncoder {
    fn blas_type() -> SparseType;
}

// generic implementation of custom
impl<T> TypeEncoder for Udf<T> {
    fn blas_type() -> SparseType {
        let grb_type: GrB_Type = grb_call(|TPE: &mut MaybeUninit<GrB_Type>| unsafe {
            GrB_Type_new(TPE.as_mut_ptr(), size_of::<Udf<T>>() as u64)
        });
        SparseType { tpe: grb_type }
    }
}

// manually define what types can act as a boolean mask (basically all basic types)
pub trait CanBool {}
impl CanBool for bool {}
impl CanBool for i32 {}

#[macro_export]
macro_rules! make_base_sparse_type {
    ( $typ:ty, $grb_typ:ident ) => {
        impl TypeEncoder for $typ {
            fn blas_type() -> SparseType {
                unsafe { SparseType { tpe: $grb_typ } }
            }
        }
    };
}

make_base_sparse_type!(bool, GrB_BOOL);
make_base_sparse_type!(i8, GrB_INT8);
make_base_sparse_type!(u8, GrB_UINT8);
make_base_sparse_type!(i16, GrB_INT16);
make_base_sparse_type!(u16, GrB_UINT16);
make_base_sparse_type!(i32, GrB_INT32);
make_base_sparse_type!(u32, GrB_UINT32);
make_base_sparse_type!(i64, GrB_INT64);
make_base_sparse_type!(u64, GrB_UINT64);
make_base_sparse_type!(f32, GrB_FP32);
make_base_sparse_type!(f64, GrB_FP64);

pub mod desc {
    extern crate num_traits;
    use crate::ops::ffi::*;
    use num_traits::{FromPrimitive, ToPrimitive};
    use std::mem::MaybeUninit;

    pub struct Descriptor {
        pub(crate) desc: GrB_Descriptor,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Primitive)]
    pub enum Field {
        Output = GrB_Desc_Field_GrB_OUTP as isize,
        Mask = GrB_Desc_Field_GrB_MASK as isize,
        Input0 = GrB_Desc_Field_GrB_INP0 as isize,
        Input1 = GrB_Desc_Field_GrB_INP1 as isize,
        AXB_Method = GrB_Desc_Field_GxB_AxB_METHOD as isize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Primitive)]
    pub enum Value {
        Default = GrB_Desc_Value_GxB_DEFAULT as isize,
        Replace = GrB_Desc_Value_GrB_REPLACE as isize,
        Structure = GrB_Desc_Value_GrB_STRUCTURE as isize,
        Transpose = GrB_Desc_Value_GrB_TRAN as isize,
        Gustavson = GrB_Desc_Value_GxB_AxB_GUSTAVSON as isize,
        Hash = GrB_Desc_Value_GxB_AxB_HASH as isize,
        Dot = GrB_Desc_Value_GxB_AxB_DOT as isize,
        SAXPY = GrB_Desc_Value_GxB_AxB_SAXPY as isize,
    }

    impl Default for Descriptor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Descriptor {
        pub fn new() -> Descriptor {
            let desc = grb_call(|D: &mut MaybeUninit<GrB_Descriptor>| unsafe {
                GrB_Descriptor_new(D.as_mut_ptr())
            });
            Descriptor { desc }
        }

        pub fn set(&mut self, key: Field, value: Value) -> &mut Descriptor {
            grb_run(|| unsafe {
                GrB_Descriptor_set(self.desc, key.to_u32().unwrap(), value.to_u32().unwrap())
            });
            self
        }

        pub fn get(&self, key: Field) -> Option<Value> {
            let value = grb_call(|X: &mut MaybeUninit<GrB_Desc_Value>| unsafe {
                GxB_Descriptor_get(X.as_mut_ptr(), self.desc, key.to_u32().unwrap())
            });
            Value::from_u32(value)
        }
    }

    impl Drop for Descriptor {
        fn drop(&mut self) {
            unsafe {
                let m_pointer = &mut self.desc as *mut GrB_Descriptor;
                GrB_Descriptor_free(m_pointer);
            }
        }
    }

    #[test]
    fn can_create_descriptor_set_field_value() {
        let mut desc = Descriptor::new();
        desc.set(Field::Mask, Value::Default);
        assert_eq!(desc.get(Field::Mask), Some(Value::Default));
    }
}
