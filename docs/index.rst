.. sprog documentation master file, created by
   sphinx-quickstart on Fri Nov  1 08:57:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sprog documentation
===================

:code:`sprog` wraps :class:`scipy.sparse.csr_array` as :class:`pandas.api.extensions.ExtensionArray`
to let you specify `Linear Programs <https://en.wikipedia.org/wiki/Linear_programming>`_ using
the familiar and readable :code:`pandas` API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

=========
extension
=========
.. automodule:: sprog.extension
.. autoclass:: sprog.extension.LinearVariable
.. autoclass:: sprog.extension.LinearVariableArray

========
function
========
.. automodule:: sprog.function
.. autofunction:: sprog.function.pos

======
sparse
======
.. automodule:: sprog.sparse
.. autofunction:: sprog.sparse.gather
.. autofunction:: sprog.sparse.scatter
.. autofunction:: sprog.sparse.repeat

