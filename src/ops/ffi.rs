#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::mem::MaybeUninit;

fn handle_grb_response(status:u32) {
    match status {
        0 => (),
        err => {
            let grb_err_text = unsafe {
                CStr::from_ptr(GrB_error()).to_str()
            };

            panic!("Error: {}, Failed to call GRB function {:?} ", err, grb_err_text);
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
