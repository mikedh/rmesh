use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn cache_access(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input as a function
    let input = parse_macro_input!(item as ItemFn);

    // Extract the function name and body
    let fn_name_ident = input.sig.ident.clone(); // The function name as an identifier
    let fn_body = input.block; // The function body
    let fn_sig = input.sig; // The function signature (parameters, return type, etc.)
    let fn_vis = input.vis; // The function visibility (e.g., `pub`)

    // Generate the expanded function with cache access logic
    let expanded = quote! {
        #fn_vis #fn_sig {
            if self._cache.read().unwrap().#fn_name_ident.is_none() {
                let temp = (|| #fn_body)();
                let mut cache = self._cache.write().unwrap();
                cache.#fn_name_ident = Some(Arc::new(temp));
            }

            self._cache
                .read()
                .unwrap()
                .#fn_name_ident
                .as_ref()
                .unwrap()
                .clone()
        }
    };

    TokenStream::from(expanded)
}