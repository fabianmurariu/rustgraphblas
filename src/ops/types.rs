#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]


use crate::ops::ffi::*;

pub struct SparseType {
    pub(crate) tpe: *mut GrB_Type,
}

pub trait TypeEncoder {
    fn blas_type() -> SparseType;
}

#[macro_export]
macro_rules! make_base_sparse_type {
    ( $typ:ty, $grb_typ:ident ) => {
        impl TypeEncoder for $typ {
            fn blas_type() -> SparseType {
                let tpe = unsafe { &mut $grb_typ as *mut GrB_Type };
                SparseType { tpe: tpe }
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

pub mod desc {
extern crate num_traits;
    use num_traits::{FromPrimitive, ToPrimitive};
    use std::mem::MaybeUninit;
    use crate::ops::ffi::*;

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
        SCMP = GrB_Desc_Value_GrB_SCMP as isize,
        Transpose = GrB_Desc_Value_GrB_TRAN as isize,
        Gustavson = GrB_Desc_Value_GxB_AxB_GUSTAVSON as isize,
        Heap = GrB_Desc_Value_GxB_AxB_HEAP as isize,
        Dot = GrB_Desc_Value_GxB_AxB_DOT as isize,
    }


    impl Default for Descriptor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Descriptor {
        pub fn new() -> Descriptor {
            let mut X = MaybeUninit::<GrB_Descriptor>::uninit();
            unsafe {
                match GrB_Descriptor_new(X.as_mut_ptr()) {
                    0 => {
                        let desc = X.as_mut_ptr();
                        Descriptor {
                            desc : *desc
                        }
                    },
                    e => panic!("Unable to create Descriptor GrB_info {}", e)
                }
            }

        }

        pub fn set(&mut self, key: Field, value: Value) -> &Descriptor {
            unsafe {
                match GrB_Descriptor_set(self.desc, key.to_u32().unwrap(), value.to_u32().unwrap()) {
                    0 => self,
                    e => panic!("Unable to set {:?}={:?} GrB_error {}", key, value, e)
                }
            }
        }

        pub fn get(&self, key:Field) -> Option<Value> {
            let mut X = MaybeUninit::<GrB_Desc_Value>::uninit();
            unsafe {
                match GxB_Descriptor_get( X.as_mut_ptr(), self.desc, key.to_u32().unwrap()) {
                    0 => Value::from_u32(X.assume_init()),
                    e => panic!("Unable to GET descriptor for {:?} GrB_error {}", key, e)
                }
            }
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
