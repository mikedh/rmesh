use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{ItemFn, ReturnType, Type, parse_macro_input, parse_quote};

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
                cache.#fn_name_ident = Some(temp);
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

#[proc_macro]
pub fn generate_inner_cache(_input: TokenStream) -> TokenStream {
    // Parse the file and extract functions with #[cache_access]
    let source_code = include_str!("../../rmesh/src/mesh.rs"); // Adjust the path if needed
    let syntax_tree = syn::parse_file(source_code).expect("Failed to parse file");

    let mut fields = Vec::new();

    for item in syntax_tree.items {
        if let syn::Item::Fn(func) = item {
            if has_cache_access(&func.attrs) {
                if let Some(return_type) = extract_return_type(&func) {
                    let field_name = format_ident!("{}", func.sig.ident);
                    fields.push(quote! {
                        #field_name: Option<#return_type>
                    });
                }
            }
        }
    }

    // Generate the InnerCache struct
    let expanded = quote! {
        #[derive(Default, Debug, Clone)]
        pub struct InnerCache {
            #(#fields),*
        }
    };

    TokenStream::from(expanded)
}

/// Check if a function has the #[cache_access] attribute
fn has_cache_access(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if let Ok(_) = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("cache_access") {
                return Ok(());
            }
            Err(meta.error("Unexpected attribute"))
        }) {
            return true;
        }
        false
    })
}

/// Extract the return type of a function
fn extract_return_type(func: &ItemFn) -> Option<Type> {
    if let ReturnType::Type(_, ty) = &func.sig.output {
        Some(*ty.clone())
    } else {
        None
    }
}

/// Macro to generate WASM functions with standardized error handling
/// Converts Rust Result types to WASM-compatible Result<T, String>
#[proc_macro_attribute]
pub fn wasm_result(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_inputs = &input.sig.inputs;
    let fn_body = &input.block;
    
    // Extract the return type and modify it to be Result<T, String>
    let return_type = match &input.sig.output {
        ReturnType::Type(_, ty) => {
            // If it's already a Result, keep it as is
            if ty.to_token_stream().to_string().contains("Result") {
                ty.clone()
            } else {
                // Wrap in Result<T, String>
                parse_quote! { Result<#ty, String> }
            }
        }
        ReturnType::Default => parse_quote! { Result<(), String> },
    };
    
    let expanded = quote! {
        #[wasm_bindgen]
        #fn_vis fn #fn_name(#fn_inputs) -> #return_type {
            (|| -> anyhow::Result<_> #fn_body)().map_err(|e| e.to_string())
        }
    };
    
    TokenStream::from(expanded)
}
