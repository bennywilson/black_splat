//! IndexedDB-backed asset cache for the wasm build (see `idb.js`).  Bytes are
//! keyed by the asset's relative path; used to cache network fetches in
//! [`load_binary`](crate::assets::load_binary) and to persist user-imported
//! models across sessions.  Every helper degrades to a no-op / empty result on
//! any JS error, so a browser without IndexedDB just falls back to plain
//! network fetches.
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen(module = "/src/idb.js")]
extern "C" {
    #[wasm_bindgen(js_name = bsIdbGet)]
    fn bs_idb_get(key: &str) -> js_sys::Promise;
    #[wasm_bindgen(js_name = bsIdbPut)]
    fn bs_idb_put(key: &str, bytes: &[u8]) -> js_sys::Promise;
    #[wasm_bindgen(js_name = bsIdbKeys)]
    fn bs_idb_keys() -> js_sys::Promise;
}

/// Cached bytes for `key`, or `None` if absent (or IndexedDB is unavailable).
pub async fn get(key: &str) -> Option<Vec<u8>> {
    let val = JsFuture::from(bs_idb_get(key)).await.ok()?;
    if val.is_null() || val.is_undefined() {
        return None;
    }
    Some(js_sys::Uint8Array::new(&val).to_vec())
}

/// Stores `bytes` under `key`.  Best effort: any error is ignored.
pub async fn put(key: &str, bytes: &[u8]) {
    let _ = JsFuture::from(bs_idb_put(key, bytes)).await;
}

/// Every stored key (empty on error).
pub async fn keys() -> Vec<String> {
    let Ok(val) = JsFuture::from(bs_idb_keys()).await else {
        return Vec::new();
    };
    js_sys::Array::from(&val)
        .iter()
        .filter_map(|v| v.as_string())
        .collect()
}
