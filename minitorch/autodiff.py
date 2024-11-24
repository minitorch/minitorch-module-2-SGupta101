from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    def _modify_arg(e):
        return [x + e if i == arg else x for i, x in enumerate(vals)]

    return (f(*_modify_arg(epsilon)) - f(*_modify_arg(-epsilon))) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()  # Set to track visited nodes by unique_id
    order = []    # List to store nodes in topological order

    def dfs(v: Variable):
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)
        # Recursively visit all parent nodes (dependencies)
        for parent in v.parents:
            if not parent.is_constant():
                dfs(parent)
        # Append the node only after all dependencies are visited
        order.append(v)

    # Start the DFS from the final variable
    dfs(variable)
    
    # Return the order in reverse since we want it from the starting nodes to the final node
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Step 1: Get topological order of variables for backpropagation
    topo_order = topological_sort(variable)

    # Step 2: Initialize derivatives dictionary, using unique_id as the key
    derivatives = defaultdict(float)
    derivatives[variable.unique_id] = deriv  # The derivative of the output node w.r.t itself is 1.0

    # Step 3: Traverse in reverse topological order to propagate gradients
    for v in topo_order:
        if v.unique_id not in derivatives:  # Skip nodes with no gradient to propagate
            continue

        current_derivative = derivatives[v.unique_id]

        # If the variable has parents, apply the chain rule to propagate derivatives
        for parent, grad in v.chain_rule(current_derivative):
            if parent.is_leaf():
                # Accumulate the derivative directly for leaf nodes
                parent.accumulate_derivative(grad)
            else:
                # Accumulate gradient for each parent node
                derivatives[parent.unique_id] += grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
