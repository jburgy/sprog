.. sprog documentation master file, created by
   sphinx-quickstart on Fri Nov  1 08:57:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sprog documentation
===================

.. toctree::

.. autosummary::
   :toctree: _autosummary
   :recursive:

   sprog
   sprog.extension
   sprog.sparse
   sprog.aggregate
   sprog.function

:mod:`sprog` is a python `Algebraic Modeling <https://en.wikipedia.org/wiki/Algebraic_modeling_language>`_
toolkit to define and solve `Linear Programs <https://en.wikipedia.org/wiki/Linear_programming>`_
using the familiar and readable :code:`pandas` API. Popular libraries like
`PuLP <https://coin-or.github.io/pulp/>`_ or `CVXPY <https://www.cvxpy.org/>`_
typically achieve this by representing linear programs as
`expression trees <https://en.wikipedia.org/wiki/Abstract_syntax_tree>`_.
These expressions require a translation step before being handed to a solver.

:mod:`sprog` approaches the problem differently based on a direct interpretation
of the `standard form <https://en.wikipedia.org/wiki/Linear_programming#Standard_form>`_
of linear problems.  Standard form matrices represent a mapping from 
*decision variables* (columns) to *objective variables* (rows).  Algebraic transformations
of *objective variables* can be expressed as
`matrix multiplications <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ with
carefully constructed matrices of coefficients.

Pretend for example that ``x`` represents a :mod:`sprog` variable of length ``n`` and
that you want to extract its elements at index ``3`` and ``7``.  This can be achieved
by::

   from scipy import sparse

   y = sparse.csr_array(([1.0, 1.0], ([0, 1], [3, 7])), shape=(2, n)) @ y

if you recall that :external+scipy:class:`scipy.sparse.csr_array` creates a ``2 × n``
matrices full of zeros except for entries ``(0, 3)`` and ``(1, 7)``.

This interpretation offers two powerful features:

#. :mod:`sprog` expressions require almost no transformation to be understood by solvers
#. once the problem is solved, multiplying a :mod:`sprog` expression with the solution vector
   returns the expression's optimal value

To manage complexity, :mod:`sprog` expressions integrate with :mod:`pandas`.
This lets users leverage :mod:`pandas`' approach to :external+pandas:ref:`indexing`.
As a practical matter, this means that :class:`sprog.extension.LinearVariableArray`
sub-classes :class:`scipy.sparse.csr_array` *and* :class:`pandas.api.extensions.ExtensionArray`.
