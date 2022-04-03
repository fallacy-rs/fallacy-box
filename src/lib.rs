//! A pointer type for heap allocation.

#![feature(allocator_api)]

pub use fallacy_clone::{AllocError, TryClone};
pub use std::alloc::{Allocator, Global, Layout};

use std::boxed::Box as StdBox;
use std::fmt;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::result::Result;

/// A pointer type for heap allocation.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Box<T: ?Sized, A: Allocator = Global>(StdBox<T, A>);

impl<T> Box<T> {
    /// Allocates memory on the heap then places `x` into it,
    /// returning an error if the allocation fails
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    #[inline]
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        Ok(Box(StdBox::try_new(x).map_err(|_| AllocError::new(Layout::new::<T>()))?))
    }
}

impl<T: ?Sized> Box<T> {
    /// Constructs a box from a raw pointer.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `Box`. Specifically, the `Box` destructor will call
    /// the destructor of `T` and free the allocated memory. For this
    /// to be safe, the memory must have been allocated in accordance
    /// with the [memory layout] used by `Box` .
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        Box(StdBox::from_raw(raw))
    }
}

impl<T, A: Allocator> Box<T, A> {
    /// Allocates memory in the given allocator then places `x` into it,
    /// returning an error if the allocation fails
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    #[inline]
    pub fn try_new_in(x: T, alloc: A) -> Result<Self, AllocError> {
        Ok(Box(
            StdBox::try_new_in(x, alloc).map_err(|_| AllocError::new(Layout::new::<T>()))?
        ))
    }
}

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Constructs a box from a raw pointer in the given allocator.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `Box`. Specifically, the `Box` destructor will call
    /// the destructor of `T` and free the allocated memory. For this
    /// to be safe, the memory must have been allocated in accordance
    /// with the [memory layout] used by `Box` .
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    #[inline]
    pub unsafe fn from_raw_in(raw: *mut T, alloc: A) -> Self {
        Box(StdBox::from_raw_in(raw, alloc))
    }

    /// Consumes the `Box`, returning a wrapped raw pointer.
    ///
    /// The pointer will be properly aligned and non-null.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory, taking
    /// into account the [memory layout] used by `Box`. The easiest way to
    /// do this is to convert the raw pointer back into a `Box` with the
    /// [`Box::from_raw`] function, allowing the `Box` destructor to perform
    /// the cleanup.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_raw(b)` instead of `b.into_raw()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    pub fn into_raw(b: Self) -> *mut T {
        StdBox::into_raw(b.0)
    }

    /// Consumes the `Box`, returning a wrapped raw pointer and the allocator.
    ///
    /// The pointer will be properly aligned and non-null.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory, taking
    /// into account the [memory layout] used by `Box`. The easiest way to
    /// do this is to convert the raw pointer back into a `Box` with the
    /// [`Box::from_raw_in`] function, allowing the `Box` destructor to perform
    /// the cleanup.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_raw_with_allocator(b)` instead of `b.into_raw_with_allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    pub fn into_raw_with_allocator(b: Self) -> (*mut T, A) {
        StdBox::into_raw_with_allocator(b.0)
    }

    /// Returns a reference to the underlying allocator.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::allocator(&b)` instead of `b.allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    pub fn allocator(b: &Self) -> &A {
        StdBox::allocator(&b.0)
    }

    /// Consumes and leaks the `Box`, returning a mutable reference,
    /// `&'a mut T`. Note that the type `T` must outlive the chosen lifetime
    /// `'a`. If the type has only static references, or none at all, then this
    /// may be chosen to be `'static`.
    ///
    /// This function is mainly useful for data that lives for the remainder of
    /// the program's life. Dropping the returned reference will cause a memory
    /// leak. If this is not acceptable, the reference should first be wrapped
    /// with the [`Box::from_raw`] function producing a `Box`. This `Box` can
    /// then be dropped which will properly destroy `T` and release the
    /// allocated memory.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::leak(b)` instead of `b.leak()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    pub fn leak<'a>(b: Self) -> &'a mut T
    where
        A: 'a,
    {
        StdBox::leak(b.0)
    }

    #[inline]
    pub fn into_std(self) -> StdBox<T, A> {
        self.0
    }

    #[inline]
    pub fn from_std(b: StdBox<T, A>) -> Self {
        Box(b)
    }
}

impl<T: ?Sized, A: Allocator> Deref for Box<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T: ?Sized, A: Allocator> DerefMut for Box<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}

impl<T: ?Sized, A: Allocator> AsRef<T> for Box<T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<T: ?Sized, A: Allocator> AsMut<T> for Box<T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.0.as_mut()
    }
}

impl<T: fmt::Display + ?Sized, A: Allocator> fmt::Display for Box<T, A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T: fmt::Debug + ?Sized, A: Allocator> fmt::Debug for Box<T, A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: ?Sized, A: Allocator> fmt::Pointer for Box<T, A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.0, f)
    }
}

impl<T: TryClone, A: Allocator + TryClone> TryClone for Box<T, A> {
    #[inline]
    fn try_clone(&self) -> Result<Self, AllocError> {
        let clone = self.0.try_clone()?;
        Ok(Box::from_std(clone))
    }

    #[inline]
    fn try_clone_from(&mut self, source: &Self) -> Result<(), AllocError> {
        self.as_mut().try_clone_from(source)
    }
}
