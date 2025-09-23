"""
Source code analyzer for static analysis of operator implementations
"""

import ast
import inspect
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from deepconstr.logger import CONSTR_LOG


class SourceAnalyzer:
    """
    Analyzes operator source code to extract constraint information
    """
    
    def __init__(self):
        self.constraint_patterns = {
            'kernel_size_constraint': r'kernel_size\s*[<>=!]+\s*.*input_size.*padding',
            'shape_constraint': r'shape.*[<>=!]+\s*\d+',
            'dimension_constraint': r'dim.*[<>=!]+\s*\d+',
            'size_constraint': r'size.*[<>=!]+\s*\d+',
            'element_constraint': r'element.*[<>=!]+\s*\d+'
        }
        
    def analyze_operator_source(self, operator_func) -> Dict[str, Any]:
        """
        Analyze operator source code to extract constraint information
        
        Args:
            operator_func: The operator function to analyze
            
        Returns:
            Dict containing extracted constraint information
        """
        try:
            source_code = inspect.getsource(operator_func)
            ast_tree = ast.parse(source_code)
            
            constraints = self._extract_constraints_from_ast(ast_tree)
            error_mappings = self._extract_error_mappings(source_code)
            
            return {
                'constraints': constraints,
                'error_mappings': error_mappings,
                'source_code': source_code
            }
        except Exception as e:
            CONSTR_LOG.warning(f"Failed to analyze operator source: {e}")
            return {}
    
    def _extract_constraints_from_ast(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extract constraint information from AST
        """
        constraints = []
        
        class ConstraintVisitor(ast.NodeVisitor):
            def visit_Compare(self, node):
                # Extract comparison constraints
                left = self._get_node_text(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    right = self._get_node_text(comparator)
                    op_text = self._get_op_text(op)
                    
                    constraints.append({
                        'type': 'comparison',
                        'left': left,
                        'operator': op_text,
                        'right': right,
                        'line': node.lineno
                    })
                self.generic_visit(node)
            
            def visit_Assert(self, node):
                # Extract assertion constraints
                test = self._get_node_text(node.test)
                constraints.append({
                    'type': 'assertion',
                    'condition': test,
                    'line': node.lineno
                })
                self.generic_visit(node)
            
            def visit_If(self, node):
                # Extract conditional constraints
                test = self._get_node_text(node.test)
                constraints.append({
                    'type': 'conditional',
                    'condition': test,
                    'line': node.lineno
                })
                self.generic_visit(node)
            
            def _get_node_text(self, node):
                """Extract text representation of AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Constant):
                    return str(node.value)
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_node_text(node.value)}.{node.attr}"
                elif isinstance(node, ast.Call):
                    func = self._get_node_text(node.func)
                    args = [self._get_node_text(arg) for arg in node.args]
                    return f"{func}({', '.join(args)})"
                else:
                    return ast.unparse(node)
            
            def _get_op_text(self, op):
                """Get text representation of comparison operator"""
                op_map = {
                    ast.Eq: '==',
                    ast.NotEq: '!=',
                    ast.Lt: '<',
                    ast.LtE: '<=',
                    ast.Gt: '>',
                    ast.GtE: '>=',
                    ast.Is: 'is',
                    ast.IsNot: 'is not',
                    ast.In: 'in',
                    ast.NotIn: 'not in'
                }
                return op_map.get(type(op), str(op))
        
        visitor = ConstraintVisitor()
        visitor.visit(ast_tree)
        return constraints
    
    def _extract_error_mappings(self, source_code: str) -> Dict[str, List[str]]:
        """
        Extract error message to constraint mappings from source code
        """
        error_mappings = {}
        
        # Pattern for error messages and their associated constraints
        error_patterns = [
            r'raise\s+\w*Error\s*\(\s*["\']([^"\']+)["\']',  # Error messages
            r'assert\s+([^,]+)',  # Assertions
            r'if\s+not\s+([^:]+):',  # Conditional checks
        ]
        
        for pattern in error_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                error_msg = match.group(1).strip()
                line_num = source_code[:match.start()].count('\n') + 1
                
                # Find associated constraint in nearby lines
                constraint = self._find_nearby_constraint(source_code, line_num)
                if constraint:
                    if error_msg not in error_mappings:
                        error_mappings[error_msg] = []
                    error_mappings[error_msg].append(constraint)
        
        return error_mappings
    
    def _find_nearby_constraint(self, source_code: str, line_num: int, context_lines: int = 5) -> Optional[str]:
        """
        Find constraint information near a specific line number
        """
        lines = source_code.split('\n')
        start_line = max(0, line_num - context_lines)
        end_line = min(len(lines), line_num + context_lines)
        
        context = lines[start_line:end_line]
        
        # Look for constraint patterns in context
        for line in context:
            for pattern_name, pattern in self.constraint_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    return line.strip()
        
        return None
    
    def analyze_torch_operator(self, operator_name: str) -> Dict[str, Any]:
        """
        Analyze PyTorch operator source code
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # Try to get the operator function
            if hasattr(torch, operator_name):
                operator_func = getattr(torch, operator_name)
            elif hasattr(F, operator_name):
                operator_func = getattr(F, operator_name)
            else:
                CONSTR_LOG.warning(f"Operator {operator_name} not found in torch or F")
                return {}
            
            return self.analyze_operator_source(operator_func)
            
        except Exception as e:
            CONSTR_LOG.warning(f"Failed to analyze torch operator {operator_name}: {e}")
            return {}
    
    def analyze_tensorflow_operator(self, operator_name: str) -> Dict[str, Any]:
        """
        Analyze TensorFlow operator source code
        """
        try:
            import tensorflow as tf
            
            # Try to get the operator function
            if hasattr(tf, operator_name):
                operator_func = getattr(tf, operator_name)
            else:
                CONSTR_LOG.warning(f"Operator {operator_name} not found in tensorflow")
                return {}
            
            return self.analyze_operator_source(operator_func)
            
        except Exception as e:
            CONSTR_LOG.warning(f"Failed to analyze tensorflow operator {operator_name}: {e}")
            return {}
