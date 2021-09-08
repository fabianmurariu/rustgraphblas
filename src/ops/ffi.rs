#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]


include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::mem::MaybeUninit;

pub type Complex<T> = __BindgenComplex<T>;

pub type fc32 = Complex<f32>;
pub type fc64 = Complex<f64>;

fn handle_grb_response(status:u32) {
    match status {
        0 => (),
        err => {
            // TODO: recover the error in a generic way
            // let grb_err_text = unsafe {
            //     CStr::from_ptr(GrB_error()).to_str()
            // };

            panic!("Error: {}, Failed to call GRB function", err);
        }
    }
}

pub fn grb_call<F, T>(mut grb_fn: F) -> T
where
    F: FnMut(&mut MaybeUninit<T>) -> u32 ,
{
    let mut P = MaybeUninit::<T>::uninit();
    handle_grb_response(grb_fn(&mut P));
    unsafe {P.assume_init()}
}

pub fn grb_run<F>(mut grb_fn: F)
where
    F: FnMut() -> u32 {
    handle_grb_response(grb_fn());
}

pub const GxB_RANGE:i64     = std::i64::MAX;
pub const GxB_STRIDE:i64    = std::i64::MAX-1;
pub const GxB_BACKWARDS:i64 = std::i64::MAX-2;


#[macro_export]
macro_rules! trait_gen_fn1{
    ( $grb_name:ident; $grb_sparse_tpe:ident; $rust_maker:ident; $( $rust_tpe:ty ),* ; $( $grb_tpe:ident ),* ) => {
        paste::item! {
            $(
                $rust_maker!(
                    $rust_tpe,
                    [<$grb_sparse_tpe _ $grb_name _ $grb_tpe>]
                );
                )*
        }
    }
}

#[macro_export]
macro_rules! trait_gen_fn0{
    ( $rust_maker:ident; $( $rust_tpe:ty ),* ) => {
            $(
                $rust_maker!(
                    $rust_tpe
                );
                )*
    }
}
