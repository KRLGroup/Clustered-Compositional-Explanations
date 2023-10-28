"""
This module is an adaptation of the formula module used
in the excellent repository https://github.com/jayelm/compexp .
It contains functions to represent a formula as a tree, to compute the hash
value of a formula and to order formulas.
"""

from functools import total_ordering
from typing import List


class F:
    """
    Abstract class to represent a formula.
    """
    pass


def compute_hash_value(formula: F) -> float:
    """
    Function to compute the hash value of a given formula.
    This function is used to avoid computation of equivalent formulas,
    thus, two equivalent formulas will have the same hash value.
    """
    if isinstance(formula, Leaf):
        return round(formula.val / 2000, 4)
    elif isinstance(formula, Not):
        return round(1 - compute_hash_value(formula.val), 4)
    elif isinstance(formula, Or):
        return round(
            compute_hash_value(formula.left)
            + compute_hash_value(formula.right)
            - (
                (
                    compute_hash_value(formula.right)
                    * compute_hash_value(formula.left)
                )
            ),
            4,
        )
    elif isinstance(formula, And):
        return round(
            compute_hash_value(
                formula.left)
            * compute_hash_value(
                formula.right), 4
        )
    return None


class Leaf(F):
    """
    Class to represent an atomic concept.
    """
    def __init__(self, val):
        self.val = val
        # We added this term to order the formulas by IoU
        self.iou = None

    def __str__(self):
        return str(self.val)

    def to_str(self, namer):
        return namer(self.val)

    def __len__(self):
        return 1

    # we redefine the hash function
    def __hash__(self):
        return hash(compute_hash_value(self))

    def __repr__(self):
        return f"Leaf({str(self)})"

    def get_vals(self):
        return [self.val]

    def is_leaf(self):
        return True

    def get_ops(self):
        return []

    # we redefine the equality function
    def __eq__(self, other):
        if isinstance(other, Not):
            if other.val == self.val:
                return False
            else:
                return other.val.val == self
        elif isinstance(other, Leaf):
            return other.val == self.val
        elif isinstance(other, BinaryNode):
            # Absorption
            if other.op == "OR" or other.op == "AND":
                if isinstance(other.left, Leaf):
                    leaf_term = other.left
                    other_term = other.right
                elif isinstance(other.right, Leaf):
                    leaf_term = other.right
                    other_term = other.left
                else:
                    return False  # no leaf term
                if leaf_term != self or not isinstance(other_term, BinaryNode):
                    return False
                if other_term.op != other.op and other_term.op != "NOT":
                    if other_term.left == self or other_term.right == self:
                        return True
                    else:
                        return False
                else:
                    return False
            return False
        else:
            return False


class Node(F):
    """
    Abstract class to represent a node in a formula.
    """
    def is_leaf(self):
        return False


class UnaryNode(Node):
    """ Class to represent a unary node. """
    arity = 1
    op = None

    def __init__(self, val):
        self.val = val
        # We added this term to order the formulas by IoU
        self.iou = 0

    def __str__(self):
        return f"({self.op} {self.val})"

    def to_str(self, namer):
        op_name = self.val.to_str(namer)
        return f"({self.op} {op_name})"

    def __len__(self):
        return len(self.val)

    def __repr__(self):
        return f"{self.op}({self.val})"

    def get_vals(self):
        return self.val.get_vals()


class BinaryNode(Node):
    """ Class to represent a binary node. """
    arity = 2
    op = None

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        # We added this term to order the formulas by IoU
        self.iou = 0

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"

    def to_str(self, namer, sort=False):
        left_name = self.left.to_str(namer, sort=sort)
        right_name = self.right.to_str(namer, sort=sort)
        if not sort or (left_name < right_name):
            return f"({left_name} {self.op} {right_name})"
        else:
            return f"({right_name} {self.op} {left_name})"

    def __len__(self):
        return len(self.left) + len(self.right)

    def __repr__(self):
        return f"{self.op}({self.left}, {self.right})"

    def get_vals(self):
        vals = []
        vals.extend(self.left.get_vals())
        vals.extend(self.right.get_vals())
        return vals

    def get_ops(self):
        """ Function to return the operators in a formula. """
        vals = []
        vals.append(self.op)
        if isinstance(self.left, BinaryNode) or isinstance(
            self.left, UnaryNode
        ):
            vals.append(self.left.op)
        if isinstance(self.right, BinaryNode) or isinstance(
            self.right, UnaryNode
        ):
            vals.append(self.right.op)
        return vals

    def __eq__(self, other):
        if isinstance(other, BinaryNode):
            right_vals = sorted(self.right.get_vals())
            left_vals = sorted(self.left.get_vals())
            other_right_vals = sorted(other.right.get_vals())
            other_left_vals = sorted(other.left.get_vals())
            right_ops = sorted(self.right.get_ops())
            left_ops = sorted(self.left.get_ops())

            other_right_ops = sorted(other.right.get_ops())
            other_left_ops = sorted(other.left.get_ops())
            if (
                (self.op == other.op)
                and (right_vals == other_right_vals)
                and (left_vals == other_left_vals)
                and (right_ops == other_right_ops)
                and (left_ops == other_left_ops)
            ):
                return True
            elif (
                (self.op == other.op)
                and (right_vals == other_left_vals)
                and (left_vals == other_right_vals)
                and (right_ops == other_left_ops)
                and (left_ops == other_right_ops)
            ):
                return True
            all_vals = sorted(right_vals + left_vals)
            other_all_vals = sorted(other_right_vals + other_left_vals)
            all_ops = sorted(right_ops + left_ops)
            other_all_ops = sorted(other_right_ops + other_left_ops)
            if (
                self.op == other.op
                and all_vals == other_all_vals
                and all_ops == other_all_ops
            ):
                return True

        elif isinstance(other, Not) and isinstance(other.val, BinaryNode):
            # De Morgan's law
            if (self.op == "OR" and other.val.op == "AND") or (
                self.op == "AND" and other.val.op == "OR"
            ):
                if other.val.left == self.left.val:
                    return other.val.right == self.right.val
                elif other.val.left == self.right.val:
                    return other.val.right == self.left.val
                else:
                    return False

        elif isinstance(other, Leaf) and isinstance(self, BinaryNode):
            # Absorption
            if self.op == "OR" or self.op == "AND":
                if isinstance(self.left, Leaf):
                    leaf_term = self.left
                    other_term = self.right
                elif isinstance(self.right, Leaf):
                    leaf_term = self.right
                    other_term = self.left
                else:
                    return False
                if leaf_term != other or not isinstance(
                    other_term, BinaryNode
                ):
                    return False
                if other_term.op != self.op and other_term.op != "NOT":
                    if other_term.left == other or other_term.right == other:
                        return True
        return False


class Not(UnaryNode):
    """
    Class to represent the NOT operator.
    """
    op = "NOT"

    def __hash__(self):
        return hash(compute_hash_value(self))

    def __eq__(self, other):
        if isinstance(other, Not):
            return self.val == other.val
        elif isinstance(self.val, Not):
            return self.val.val == other
        elif isinstance(other, Leaf):
            return self.val == other.val
        elif isinstance(other, BinaryNode) and isinstance(
            self.val, BinaryNode
        ):
            # De Morgan's law
            if (other.op == "OR" and self.val.op == "AND") or (
                other.op == "AND" and self.val.op == "OR"
            ):
                if self.val.left == other.left.val:
                    return self.val.right == other.right.val
                elif self.val.left == other.right.val:
                    return self.val.right == other.left.val
                else:
                    return False
        else:
            return False

    def get_ops(self):
        """ Function to return the operators in a formula. """
        vals = []
        vals.append(self.op)
        if isinstance(self.val, BinaryNode) or isinstance(self.val, UnaryNode):
            vals.append(self.val.op)
        if isinstance(self.val, BinaryNode) or isinstance(self.val, UnaryNode):
            vals.append(self.val.op)
        return vals


class Or(BinaryNode):
    """Class to represent the OR operator."""
    op = "OR"

    def __hash__(self):
        return hash(compute_hash_value(self))


class And(BinaryNode):
    """Class to represent the AND operator."""
    op = "AND"

    def __hash__(self):
        return hash(compute_hash_value(self))


@total_ordering
class OrderedFormula:
    """
    Class to order formulas by their iou.
    """
    def __init__(self, formula) -> None:
        self.formula = formula
        self.iou = formula.iou

    def __eq__(self, __o: object) -> bool:
        return self.iou == __o.iou

    def __lt__(self, __o: object) -> bool:
        return self.iou < __o.iou


def get_formula_str(f: F, namer_vec: List[str]):
    """
    Function to get the string representation of a formula.

    Args:
        f: Formula to get the string representation of.
        namer_vec: List of names for the variables in the formula.

    Returns:
        String representation of the formula.
    """
    if isinstance(f, And):
        masks_l = get_formula_str(f.left, namer_vec)
        masks_r = get_formula_str(f.right, namer_vec)
        return f"({masks_l} AND {masks_r})"
    elif isinstance(f, Or):
        masks_l = get_formula_str(f.left, namer_vec)
        masks_r = get_formula_str(f.right, namer_vec)
        return f"({masks_l} OR {masks_r})"
    elif isinstance(f, Not):
        return f"NOT {get_formula_str( f.val, namer_vec)}"
    elif isinstance(f, Leaf):
        return namer_vec[f.val]
    elif isinstance(f, int):
        return namer_vec[f]
