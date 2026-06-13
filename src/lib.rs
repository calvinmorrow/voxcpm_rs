pub mod audio_utils;
pub mod audio_vae_v2;
pub mod audiovae;
pub mod minicpm4;
pub mod openai_error;
pub mod openai_types;
pub mod voice_registry;
pub mod voxcpm;

pub fn type_name_of_function<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

#[macro_export]
macro_rules! vdbg_inner {
    ($val:expr $(,)?) => {
        match $val {
            tmp => {
                eprint!("({:?}, {:?}), ", &tmp.dims(), &tmp.dtype());
                tmp
            }
        }
    };
    ($($val:expr),+ $(,)?) => {
        ($($crate::vdbg_inner!($val)),+,)
    };
}

#[macro_export]
macro_rules! vdbg {
    ($($all:tt)*) => {
        fn f() {}
        let mut chunks = $crate::type_name_of_function(f).split("<_>");
        let struct_name = chunks.next().unwrap().split("::").last().unwrap();
        let mut n_chunks = chunks.next().unwrap().split("::");
        n_chunks.next();
        let function_name = n_chunks.next().unwrap();
        if function_name == "forward_step" {
            eprint!("{}::forward_step", struct_name);
        } else {
            eprint!("{}: ", struct_name);
        }
        $crate::vdbg_inner!($($all)*);
        eprintln!();
    };
}
