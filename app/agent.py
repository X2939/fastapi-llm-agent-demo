import json
import re
from typing import Any

from app.llm_client import chat_completion
from app.schemas import AgentChatResponse
from app.tools import calculator, file_summary, sql_query


TOOL_SELECTION_PROMPT = """
You are a tool-calling agent. Select exactly one tool for the user request.

Available tools:
1. calculator
   - Use for arithmetic calculation.
   - Arguments: {"expression": "a math expression"}

2. file_summary
   - Use when the user wants to read or summarize a local text file.
   - Arguments: {"file_path": "data/sample.txt"}

3. sql_query
   - Use when the user wants to query product data from a SQLite database.
   - Table: products(id, name, category, price, stock)
   - Arguments: {"query": "SELECT ... FROM products ..."}
   - Only SELECT queries are allowed.

Return JSON only. Do not return markdown. Do not explain.
JSON format:
{
  "tool": "calculator | file_summary | sql_query",
  "arguments": {}
}
"""


def run_agent(user_message: str) -> AgentChatResponse:
    tool_call = _select_tool_with_llm(user_message)# 让AI选工具
    selected_tool = tool_call["tool"]# # 拿到工具名
    tool_arguments = tool_call["arguments"] # 拿到对应的参数

    try:
        tool_result = _run_tool(selected_tool, tool_arguments)# 运行工具，拿到结果
    except Exception as exc:
        error_message = f"Tool execution failed:{exc}"

        return AgentChatResponse(
            user_message=user_message,
            selected_tool=selected_tool,
            tool_arguments=tool_arguments,
            tool_result=error_message,
            final_answer="抱歉，工具执行失败了。请检查输入或者工具参数！",
        )
    
    return AgentChatResponse(
        user_message=user_message,
        selected_tool=selected_tool,
        tool_arguments=tool_arguments,
        tool_result=tool_result,
        final_answer=_build_final_answer(
            user_message=user_message,
            selected_tool=selected_tool,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
        )
    )


def _select_tool_with_llm(user_message: str) -> dict[str, Any]:
    llm_output = chat_completion(
        messages=[
            {"role": "system", "content": TOOL_SELECTION_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    try:
        tool_call = _parse_json_object(llm_output)
    except ValueError:
        return _fallback_select_tool(user_message)

    tool = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})

    if tool not in {"calculator", "file_summary", "sql_query"}:
        return _fallback_select_tool(user_message)

    if not isinstance(arguments, dict):
        return _fallback_select_tool(user_message)

    arguments = _normalize_tool_arguments(tool, arguments)

    return {"tool": tool, "arguments": arguments}


def _normalize_tool_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool == "file_summary":
        return {"file_path": "data/sample.txt"}

    return arguments


def _parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")

    return json.loads(text[start : end + 1])


def _run_tool(selected_tool: str, arguments: dict[str, Any]) -> Any:
#❌ 错误（你以为的）
#python
#运行
#"arguments": "3+1"
#✅ 正确（代码必须要的）
#python
#运行
#"arguments": {"expression": "3+1"}
    if selected_tool == "calculator":
        return calculator(arguments["expression"])

    if selected_tool == "file_summary":
        file_path = arguments.get("file_path", "data/sample.txt")
        return file_summary(file_path)

    if selected_tool == "sql_query":
        return sql_query(arguments["query"])

    raise ValueError(f"Unknown tool: {selected_tool}")


def _build_final_answer(
    user_message:str,
    selected_tool:str,
    tool_arguments:dict[str, Any],
    tool_result:Any,
    ) -> str:

    messages=[{"role":"system",
               "content":(
                   "You are an AI Agent response generator. "
                   "You must answer the user based only on tne tool result."
                   "Do not invent facts.Keep the answer cocise and clear. "
                   ),
                },
                 {
                "role": "user",
                "content": (
                    f"用户原始问题：{user_message}\n"
                    f"选择的工具：{selected_tool}\n"
                    f"工具参数：{json.dumps(tool_arguments, ensure_ascii=False)}\n"
                    f"工具执行结果：{json.dumps(tool_result, ensure_ascii=False)}\n\n"
                    "请基于工具执行结果，用中文给出最终回答。"
                    ),
                },
            ]
    
    return chat_completion(messages=messages, temperature=0.2)


def _fallback_select_tool(user_message: str) -> dict[str, Any]:
    if "文件" in user_message or "总结" in user_message or "summary" in user_message.lower():
        return {
            "tool": "file_summary",
            "arguments": {"file_path": "data/sample.txt"},
        }

    if "查询" in user_message or "数据库" in user_message or "sql" in user_message.lower():
        return {
            "tool": "sql_query",
            "arguments": {
                "query": "SELECT id, name, category, price, stock FROM products WHERE price > 100"
            },
        }

    return {
        "tool": "calculator",
        "arguments": {"expression": _extract_math_expression(user_message)},
    }


def _extract_math_expression(text: str) -> str:
    matches = re.findall(r"[0-9+\-*/().\s]+", text)
    expression = "".join(matches).strip()

    if not expression:
        raise ValueError("No math expression found in user message.")

    return expression
