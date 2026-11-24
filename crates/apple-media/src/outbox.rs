//! Output callbacks as channels.
//!
//! VideoToolbox hands each decoded or encoded picture to a C callback, on a thread it chooses (always its own for
//! compression; usually, not always, the caller's for decompression), with a context pointer we gave it. Rather than
//! share mutable state with that callback, each session owns an `Outbox`: the callback only *sends* its result on a
//! channel, and the session's owner drains the channel when it wants output. Nothing is shared but the `Sender`,
//! which is made for exactly that, so there is no lock to forget and no aliasing to get wrong.
//!
//! The one unavoidable `unsafe` is in `sender`: turning the context pointer back into a reference.

use std::ffi::c_void;
use std::sync::mpsc::{Receiver, Sender, channel};

pub(crate) struct Outbox<T> {
    /// Boxed so its address, which the session holds, never moves.
    tx: Box<Sender<T>>,
    rx: Receiver<T>,
}

impl<T> Outbox<T> {
    pub(crate) fn new() -> Self {
        let (tx, rx) = channel();
        Self { tx: Box::new(tx), rx }
    }

    /// The context pointer for the callback (see `sender`). It stays valid as long as this outbox; the session
    /// using it must be invalidated before the outbox is dropped.
    pub(crate) fn context(&self) -> *mut c_void {
        (&raw const *self.tx).cast_mut().cast()
    }

    /// Everything the callback has sent so far.
    pub(crate) fn drain(&self) -> Vec<T> {
        self.rx.try_iter().collect()
    }
}

/// The callback's side: the sender behind a context pointer from `Outbox::context`.
///
/// # Safety
/// `context` must come from `Outbox::<T>::context` (same `T`) of an outbox that is still alive.
pub(crate) unsafe fn sender<'a, T>(context: *mut c_void) -> &'a Sender<T> {
    // SAFETY: by the contract above, a live, boxed Sender<T>; only shared access is taken.
    unsafe { &*(context as *const Sender<T>) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_from_another_thread_arrive_in_order() {
        let outbox = Outbox::<u32>::new();
        let context = outbox.context() as usize; // as a C library would hold it
        std::thread::spawn(move || {
            // SAFETY: the outbox outlives this thread (joined below).
            let tx = unsafe { sender::<u32>(context as *mut c_void) };
            for i in 0..1000 {
                tx.send(i).unwrap();
            }
        })
        .join()
        .unwrap();
        assert_eq!(outbox.drain(), (0..1000).collect::<Vec<_>>());
        assert!(outbox.drain().is_empty());
    }
}
