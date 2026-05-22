import ast
import math

class CodeComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.try_count = 0
        self.nesting_depth = 0
        self.max_nesting = 0
        
    def visit_Try(self, node):
        self.try_count += 1
        self.generic_visit(node)
        
    def visit_If(self, node):
        self.nesting_depth += 1
        self.max_nesting = max(self.max_nesting, self.nesting_depth)
        self.generic_visit(node)
        self.nesting_depth -= 1

def analyze_ast(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"error": "Invalid Python code"}
        
    visitor = CodeComplexityVisitor()
    visitor.visit(tree)
    
    # Calculate Halstead Difficulty Proxy
    operators = set(['Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'FloorDiv'])
    operands = set([node.id for node in ast.walk(tree) if isinstance(node, ast.Name)])
    
    n1 = len(operators)
    n2 = len(operands)
    difficulty = (n1 / 2) * (n2 / max(1, len(operands))) # simplified
    
    return {
        "try_catch_blocks": visitor.try_count,
        "max_nesting_depth": visitor.max_nesting,
        "halstead_difficulty_proxy": difficulty
    }
