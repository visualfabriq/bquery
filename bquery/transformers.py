from ast import NodeTransformer
from ast import (Eq, NotEq, In, NotIn, BitOr, BitAnd, 
                 Compare, BinOp, 
                 Name, Load, Str,
                 )
import ast
import copy

import bcolz

__all__ = ['standard_transformers',
           'QueryTransformer',
           'InOperatorTransformer',
           'CachedFactorOptimizer',
           'TrivialBooleanExpressionsOptimizer',
           ]

class QueryTransformer(NodeTransformer):
    """A :class:`ast.NodeTransformer` subclass that walks the abstract syntax tree 
    of the query and allows modification of nodes.
    
    The user-provided dictionary of the variables in expression that is passed 
    to the `bcolz.eval()` function can accessed and modified as 
    :attr:`self.user_dict`. The calling :class:`ctable` instance is available 
    as :attr:`self.ctable_`.

    The `QueryTransformer` will walk the AST and use the return value of the
    visitor methods to replace or remove the old node. If the return value of
    the visitor method is ``None``, the node will be removed from its location,
    otherwise it is replaced with the return value. The return value may be the
    original node in which case no replacement takes place.

    Keep in mind that if the node you're operating on has child nodes you must
    either transform the child nodes yourself or call the :meth:`generic_visit`
    method for the node first.

    For nodes that were part of a collection of statements (that applies to all
    statement nodes), the visitor may also return a list of nodes rather than
    just a single node.

    The visitor functions for the nodes are ``'visit_'`` + class name of the 
    node. So a `TryFinally` node visit function would be `visit_TryFinally`. 
    If no visitor function exists for a node (return value `None`) the 
    `generic_visit` visitor is used instead.

    Usually you use the transformer like this::
    node = QueryTransformer().apply(ctable_, node, user_dict)
    """

    def apply(self, ctable_, node, user_dict):
        self.user_dict = user_dict
        self.ctable_ = ctable_
        return self.visit(node)


class InOperatorTransformer(QueryTransformer):
    """A :class:`QueryTransformer` that converts comparisons with `in` and 
    `not in` operators into expressions using `==` and `!=`.

    Example:
        `my_col in ['ABC', 'DEF']` is transformed into
        `(my_col == 'ABC') | (my_col == 'DEF')`
    
    This is useful as Numexpr currently does not support `in` operators."""

    def visit_Compare(self, node):
        # first transform all child nodes if necessary
        node = self.generic_visit(node)

        if not isinstance(node.ops[0], (In, NotIn)):
            return node

        # replace `in` comparisions with empty comparison list
        if len(node.comparators[0].elts) == 0:
            if isinstance(node.ops[0], In):
                return Name(id='False', ctx=Load())
            else:
                return Name(id='True', ctx=Load())

        compare_op , binop_op = self.get_operators(node.ops[0])
        # rewrite the first element in list using `==` / `!=` comparison
        eq_expr = Compare(
            left = node.left, 
            ops = [compare_op], 
            comparators = [node.comparators[0].elts[0]])
        # join similar comparisons for all othe elements using the appropriate
        # binary operator, i.e. | or &
        for element in node.comparators[0].elts[1:]:
            eq_expr = BinOp(
                left = eq_expr,
                op = binop_op,
                right = Compare(
                    left = copy.copy(node.left),
                    ops = [compare_op],
                    comparators = [element]
                    )
                )
        return eq_expr

    def get_operators(self, op):
        if isinstance(op, In):
            return Eq(), BitOr()
        else:
            return NotEq(), BitAnd()


class CachedFactorOptimizer(QueryTransformer):
    """A :class:`QueryTransformer` that converts comparisons containing 
    columns with cached factors into comparisons using the factor instead.

    This potentially speeds up queries significantly:
     - By detecting queries that will not return any values without 
       scanning the entire column.
     - By evaluating the comparison on the integer typed factor rather than
       a column of a datatype that is more costly to compare, e.g. String.
       
    The `CachedFactorOptimizer` should be followed by the 
    :class:`TrivialBooleanExpressionsOptimizer` to obtain the full benefit."""

    def visit_Compare(self, node):
        # first transform all child nodes if necessary
        node = self.generic_visit(node)

        # check we have a simple comparison
        if len(node.comparators) != 1 or len(node.ops) != 1:
            return node

        # TODO: we currently do not sort the values of the cached
        #       factors. Therefore we cannot optimize inequalities
        elif not isinstance(node.ops[0], (Eq, NotEq)):
            return node

        # col_name == 'value'
        if isinstance(node.left, Name):
            var = node.left
            val = node.comparators[0]
        # 'value' == col_name
        elif isinstance(node.comparators[0], Name):
            var = node.comparators[0]
            val = node.left
        # we can accelerate expressions that contain at least one column ref
        else:
            return node

        col = var.id
        if not self.ctable_.cache_valid(col):
            return node

        # find factor id for requested value
        col_values_rootdir = self.ctable_[col].rootdir + '.values'
        carray_values = bcolz.carray(rootdir=col_values_rootdir, 
                                        mode='r')
        idx = None
        # deal with strings and number nodes
        val_field = 's' if isinstance(val, Str) else 'n'
        for index, value in enumerate(carray_values.iter()):
            if value == getattr(val, val_field):
                idx = index
                break
        # value not in cached factorisation
        if idx is None:
            if isinstance(node.ops[0], Eq):
                return Name(id='False', ctx=Load())
            else:
                return Name(id='True', ctx=Load())

        # found value in cached factorisation:
        # rewrite the comparison expression
        setattr(val, val_field, idx)
        var.id = 'bquery_factors_%s' % col
        # load the factor for later use
        if not self.user_dict.has_key('bquery_factors_%s' % col):
            col_factor_rootdir = self.ctable_[col].rootdir + '.factor'
            self.user_dict['bquery_factors_%s' % col] = \
                bcolz.carray(rootdir=col_factor_rootdir, mode='r')
        return node


class TrivialBooleanExpressionsOptimizer(QueryTransformer):
    """A :class:`QueryTransformer` that simplifies boolean expression 
    containing subparts that are trivial boolean expressions.

    Example:
        `(my_col == 'ABC') | (False)` is transformed into
        `False`

    This speeds up queries that can be logically determined to never return 
    any entries are not explicitly evaluated against the database."""

    def visit_BinOp(self, node):
        # first transform all child nodes if necessary
        node = self.generic_visit(node)

        # only optimize & and | expressions
        if not isinstance(node.op, (BitOr, BitAnd)):
            return node

        if isinstance(node.left, Name):
            name_operand = node.left
            other_operand = node.right
        elif isinstance(node.right, Name):
            name_operand = node.right
            other_operand = node.left
        # no Name operand means no trivial boolean expressions
        else:
            return node

        # the Name operand is not a trivial boolean expression but a variable
        if name_operand.id not in ['True', 'False']:
            return node

        # simplify comparisons containing trivial boolean expression
        if isinstance(node.op, BitOr):
            if name_operand.id == 'True':
                return name_operand
            else:
                return other_operand
        else:
            if name_operand.id == 'False':
                return name_operand
            else:
                return other_operand

# provides a convenient short-cut for configuring a set of standard transformers
standard_transformers = [InOperatorTransformer(),
                         TrivialBooleanExpressionsOptimizer(),
                         ]
