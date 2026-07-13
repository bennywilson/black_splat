// IndexedDB-backed asset store for the wasm build (see src/idb.rs).  Persists
// downloaded and user-imported asset bytes (models/textures/splats) so reloads
// don't refetch and web imports survive across sessions.  Values are keyed by
// the asset's relative path (e.g. "game_assets/models/quad.glb").
const DB_NAME = "black_splat_assets";
const STORE = "assets";

function openDb() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

export async function bsIdbGet(key) {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
        const req = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
        req.onsuccess = () => resolve(req.result ? new Uint8Array(req.result) : null);
        req.onerror = () => reject(req.error);
    });
}

export async function bsIdbPut(key, bytes) {
    const db = await openDb();
    // Copy out of wasm linear memory into a standalone ArrayBuffer: the view
    // handed in by wasm-bindgen would detach if memory grew mid-transaction.
    const buf = bytes.slice().buffer;
    return await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(buf, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

export async function bsIdbKeys() {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
        const req = db.transaction(STORE, "readonly").objectStore(STORE).getAllKeys();
        req.onsuccess = () => resolve(req.result.map(String));
        req.onerror = () => reject(req.error);
    });
}
