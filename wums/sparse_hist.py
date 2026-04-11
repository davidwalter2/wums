"""Sparse histogram wrapper combining a scipy sparse array with hist axes.

The :class:`SparseHist` class pairs a scipy sparse array with a sequence of
hist axes describing its dense N-D shape. The dense layout is always the
with-flow layout (each axis contributes ``axis.extent`` bins). Consumers can
extract either the with-flow or no-flow representation via the ``flow``
parameter on :meth:`SparseHist.toarray` and :meth:`SparseHist.to_flat_csr`.
"""

import numpy as np


class _AxesTuple(tuple):
    """Tuple of hist axes that supports lookup by name as well as by index."""

    def __getitem__(self, key):
        if isinstance(key, str):
            for ax in self:
                if ax.name == key:
                    return ax
            raise KeyError(f"axis '{key}' not found")
        return tuple.__getitem__(self, key)


class SparseHist:
    """Wrapper combining a scipy sparse array with hist axes describing its dense shape.

    The dense N-D layout is **always the with-flow layout**: each axis contributes
    ``axis.extent`` bins, where (for axes with underflow) position 0 is the
    underflow bin, regular bins follow at positions 1..size, and (for axes with
    overflow) the overflow bin is at the last position. For axes that have
    neither underflow nor overflow, ``extent == size`` and the layout matches the
    no-flow layout exactly.

    The user provides scipy sparse data whose row-major flattening matches the
    row-major flattening of this with-flow dense shape. Consumers (such as the
    rabbit ``TensorWriter``) can extract either the with-flow or no-flow layout
    via the ``flow`` parameter on :meth:`toarray` and :meth:`to_flat_csr`.

    Parameters
    ----------
    data : scipy.sparse array or matrix
        Sparse storage. Total element count must equal the product of axis extents.
    axes : sequence of hist axes
        Axes describing the dense N-D shape. Each axis must have ``.name``.
    metadata : optional
        Arbitrary user metadata, accessible (and assignable) via the
        ``.metadata`` attribute. Defaults to ``None``, matching the
        ``hist.Hist`` interface.
    """

    @staticmethod
    def _underflow_offset(ax):
        """Return 1 if the axis has an underflow bin, 0 otherwise."""
        traits = getattr(ax, "traits", None)
        if traits is not None and getattr(traits, "underflow", False):
            return 1
        return 0

    def __init__(self, data, axes, *, metadata=None):
        self._axes = _AxesTuple(axes)
        self._dense_shape = tuple(int(a.extent) for a in self._axes)
        self._size = int(np.prod(self._dense_shape))
        self.metadata = metadata

        if not (hasattr(data, "toarray") and hasattr(data, "tocoo")):
            raise TypeError(
                f"data must be a scipy sparse array/matrix, got {type(data).__name__}"
            )

        if int(np.prod(data.shape)) != self._size:
            raise ValueError(
                f"Total elements in sparse data ({int(np.prod(data.shape))}) does "
                f"not match product of axis extents {self._dense_shape} = {self._size}"
            )

        # Internally store as flat (indices, values) corresponding to row-major
        # flatten of the with-flow dense shape.
        coo = data.tocoo()
        if coo.ndim == 2:
            flat_idx = np.ravel_multi_index((coo.row, coo.col), data.shape)
        elif coo.ndim == 1:
            flat_idx = coo.coords[0]
        else:
            raise ValueError(f"Unsupported sparse ndim {coo.ndim}")

        self._flat_indices = np.asarray(flat_idx, dtype=np.int64)
        self._values = np.asarray(coo.data)

    @classmethod
    def _from_flat(cls, flat_indices, values, axes, size, metadata=None):
        """Construct directly from flat indices and values, bypassing __init__ checks."""
        obj = cls.__new__(cls)
        obj._axes = _AxesTuple(axes)
        obj._dense_shape = tuple(int(a.extent) for a in obj._axes)
        obj._size = int(size)
        obj._flat_indices = np.asarray(flat_indices, dtype=np.int64)
        obj._values = np.asarray(values)
        obj.metadata = metadata
        return obj

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self._dense_shape

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def nnz(self):
        return len(self._flat_indices)

    def toarray(self, flow=True):
        """Return the dense N-D numpy array.

        If ``flow=True`` (default), the result has the with-flow shape (extents).
        If ``flow=False``, flow bins are dropped and the result has the no-flow
        shape (sizes).
        """
        out = np.zeros(self._size, dtype=self._values.dtype)
        out[self._flat_indices] = self._values
        full = out.reshape(self._dense_shape)
        if flow:
            return full
        slices = tuple(
            slice(self._underflow_offset(ax), self._underflow_offset(ax) + len(ax))
            for ax in self._axes
        )
        return full[slices]

    def tocoo(self):
        """Return a 2D scipy COO array of shape (1, size) in the with-flow layout."""
        import scipy.sparse

        return scipy.sparse.coo_array(
            (
                self._values,
                (np.zeros(len(self._flat_indices), dtype=np.int64), self._flat_indices),
            ),
            shape=(1, self._size),
        )

    def to_flat_csr(self, dtype, flow=True):
        """Return a flat CSR array of shape (1, size) with sorted indices.

        If ``flow=True`` (default), returns the with-flow CSR (size = product of
        extents). If ``flow=False``, drops entries that fall in flow bins and
        returns a CSR in the no-flow layout (size = product of sizes), with
        indices shifted to that layout.
        """
        import scipy.sparse

        if flow:
            target_size = self._size
        else:
            no_flow_shape = tuple(int(len(ax)) for ax in self._axes)
            target_size = int(np.prod(no_flow_shape))

        # Use int64 indices when the flat size exceeds the int32 range, since
        # scipy.sparse CSR indices default to int32 and would silently overflow.
        idx_dtype = np.int64 if target_size > np.iinfo(np.int32).max else np.int32

        if flow:
            sort_order = np.argsort(self._flat_indices)
            sorted_idx = self._flat_indices[sort_order].astype(idx_dtype)
            sorted_vals = self._values[sort_order].astype(dtype)
            indptr = np.array([0, len(sorted_vals)], dtype=idx_dtype)
            return scipy.sparse.csr_array(
                (sorted_vals, sorted_idx, indptr), shape=(1, self._size)
            )

        # No-flow extraction: filter entries in flow bins, shift remaining to
        # the no-flow layout.
        if len(self._flat_indices) == 0:
            indptr = np.array([0, 0], dtype=idx_dtype)
            return scipy.sparse.csr_array(
                (
                    np.zeros(0, dtype=dtype),
                    np.zeros(0, dtype=idx_dtype),
                    indptr,
                ),
                shape=(1, target_size),
            )

        multi = np.unravel_index(self._flat_indices, self._dense_shape)
        mask = np.ones(len(self._flat_indices), dtype=bool)
        for i, ax in enumerate(self._axes):
            u = self._underflow_offset(ax)
            s = int(len(ax))
            mask &= (multi[i] >= u) & (multi[i] < u + s)

        shifted = tuple(
            multi[i][mask] - self._underflow_offset(ax)
            for i, ax in enumerate(self._axes)
        )

        if len(no_flow_shape) == 1:
            new_flat = shifted[0]
        else:
            new_flat = np.ravel_multi_index(shifted, no_flow_shape)

        new_values = self._values[mask]
        sort_order = np.argsort(new_flat)
        sorted_idx = new_flat[sort_order].astype(idx_dtype)
        sorted_vals = new_values[sort_order].astype(dtype)
        indptr = np.array([0, len(sorted_vals)], dtype=idx_dtype)
        return scipy.sparse.csr_array(
            (sorted_vals, sorted_idx, indptr), shape=(1, target_size)
        )

    def __mul__(self, other):
        """Multiply all stored values by a scalar, returning a new SparseHist."""
        if not isinstance(other, (int, float, np.integer, np.floating)):
            return NotImplemented
        return SparseHist._from_flat(
            self._flat_indices,
            self._values * other,
            self._axes,
            self._size,
            metadata=self.metadata,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        """In-place scalar multiplication."""
        if not isinstance(other, (int, float, np.integer, np.floating)):
            return NotImplemented
        self._values[...] *= other
        return self

    def __getitem__(self, slice_dict):
        """Slice along one or more axes by integer index, returning a new SparseHist.

        Slice indices are interpreted as regular-bin indices (0..axis.size-1),
        matching hist's ``h[{"name": i}]`` convention. The underflow offset is
        added internally so the slice maps to the correct position in the
        with-flow dense layout.
        """
        if not isinstance(slice_dict, dict):
            raise TypeError(
                f"SparseHist supports only dict-style index slicing, got {type(slice_dict).__name__}"
            )

        slice_per_axis = {}
        for ax_name, ax_idx in slice_dict.items():
            try:
                ax_pos = next(i for i, a in enumerate(self._axes) if a.name == ax_name)
            except StopIteration as ex:
                raise KeyError(
                    f"Axis '{ax_name}' not found in SparseHist axes "
                    f"{[a.name for a in self._axes]}"
                ) from ex
            ax = self._axes[ax_pos]
            slice_per_axis[ax_pos] = int(ax_idx) + self._underflow_offset(ax)

        keep_positions = [i for i in range(len(self._axes)) if i not in slice_per_axis]
        if not keep_positions:
            raise ValueError("Cannot slice all axes of a SparseHist")
        axes_keep = [self._axes[i] for i in keep_positions]

        # Convert flat indices back to multi-dim
        multi = np.unravel_index(self._flat_indices, self._dense_shape)

        # Filter entries that match the requested slice
        mask = np.ones(len(self._flat_indices), dtype=bool)
        for ax_pos, sl in slice_per_axis.items():
            mask &= multi[ax_pos] == sl

        new_dense_shape = tuple(int(a.extent) for a in axes_keep)
        new_size = int(np.prod(new_dense_shape))

        if len(keep_positions) == 1:
            new_flat = multi[keep_positions[0]][mask]
        else:
            new_multi = tuple(multi[i][mask] for i in keep_positions)
            new_flat = np.ravel_multi_index(new_multi, new_dense_shape)

        new_values = self._values[mask]
        return SparseHist._from_flat(
            new_flat, new_values, axes_keep, new_size, metadata=self.metadata
        )
