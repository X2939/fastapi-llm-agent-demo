import ast
import operator
import sqlite3

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def calculator(expression: str) -> float:
    """
    Safely evaluate a simple expression.
    Supports +,-,/,*,** and parentheses.
    """

    tree = ast.parse(expression,mode="eval")
    return float(_eval_ast_node(tree.body))

def file_summary(file_path:str,max_chars:int = 300)->str:
    """
    Read a local text file and return a short rule-based summary.
    """
    with open(file_path,"r",encoding="utf-8")as file:
        content = file.read()

    content = content.strip()#去掉前后空格、换行

    if not content:
        return "The file is empty."
    
    if len(content) <=max_chars:
        return content 
    
    return content[:max_chars] + "..."

def _eval_ast_node(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):#1.如果该树只是数字，则直接返回该树的值
        return node.value

    if isinstance(node,ast.BinOp):#2.如果该树是二元运算符，则递归调用该树
        operator_type = type(node.op)
        if operator_type not in _ALLOWED_OPERATORS:#检查该树的运算符是否在允许的运算符列表中，否则抛出异常
            raise ValueError(f"Unsupported operator:{operator_type.__name__}")
    
        left=_eval_ast_node(node.left)#递归调用该树的左子树
        right = _eval_ast_node(node.right)#递归调用该树的右子树
        return _ALLOWED_OPERATORS[operator_type](left,right)# 递归出口

    if isinstance(node,ast.UnaryOp):#3.如果该树是单目运算符
        operator_type = type(node.op)
        if operator_type not in _ALLOWED_OPERATORS:
             raise ValueError(f"Unsupported unary operator:{operator_type.__name__}")

        operand = _eval_ast_node(node.operand)#递归该操作符的孩子，由于单目运算符只有一个孩子，但孩子不一定是一个数，例如-(3*5)，对于-需要递归调用该孩子
        return _ALLOWED_OPERATORS[operator_type](operand)# 递归出口

    raise ValueError("Only simple arithmetic expressions are allowed.")


def sql_query(query:str,db_path:str="data/demo.db")->list[dict]:
    """
    Execute a read -only SQL query against the demo SQLite database.
    """
    normalized_query = query.strip().lower()

    if not normalized_query.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
