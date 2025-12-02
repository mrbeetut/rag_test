import os
import re
import json
import logging
import math
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from enum import Enum

import wikipedia
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# --- config ---
load_dotenv()

class AppConfig(BaseSettings):
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    llm_model: str = Field("openai/gpt-4o", env="LLM_MODEL")
    app_url: str = Field("http://localhost", env="APP_URL")
    app_title: str = Field("ReActAgent", env="APP_TITLE")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

config = AppConfig()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Agent")

# --- DATA MODELS (A2A Protocol) ---
class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class AgentPayload(BaseModel):
    answer: str
    reasoning_trace: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    """std answer (A2A)"""
    status: AgentStatus
    protocol_version: str = "1.0.0"
    payload: Optional[AgentPayload] = None
    error_message: Optional[str] = None

# --- TOOLS ABSTRACTION ---
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    @abstractmethod
    def description(self) -> str: pass

    @abstractmethod
    def run(self, query: str) -> str: pass

class WikipediaTool(BaseTool):
    name = "wikipedia"
    description = "Поиск информации в энциклопедии. Принимает поисковый запрос."

    def __init__(self, lang: str = 'ru'):
        wikipedia.set_lang(lang)

    def run(self, query: str) -> str:
        try:
            # Сначала ищем варианты, чтобы избежать DisambiguationError
            search_res = wikipedia.search(query)
            if not search_res:
                return "Info: Ничего не найдено."
            
            # Берем первую страницу
            page = wikipedia.page(search_res[0], auto_suggest=False)
            return f"Page: {page.title}\nSummary: {page.summary[:1000]}..."
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Error: Неоднозначность. Варианты: {', '.join(e.options[:5])}"
        except Exception as e:
            logger.error(f"Wiki error: {e}")
            return "Error: Ошибка при чтении Википедии."

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Выполнение точных математических расчетов."

    def run(self, query: str) -> str:
        allowed = set("0123456789.+-*/() ")
        if not set(query).issubset(allowed):
            return "Error: Недопустимые символы в выражении."
        try:
            # pylint: disable=eval-used
            result = eval(query, {"__builtins__": None}, {})
            return str(round(result, 4))
        except Exception as e:
            return f"Error: {e}"

# --- 4. AGENT LOGIC ---
class OpenRouterAgent:
    def __init__(self, tools: List[BaseTool]):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key,
        )
        self.model = config.llm_model
        self.tools = {t.name: t for t in tools}
        self.max_steps = 8

    def _get_headers(self):
        """Заголовки, требуемые OpenRouter"""
        return {
            "HTTP-Referer": config.app_url,
            "X-Title": config.app_title,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _llm_call(self, messages: List[Dict]) -> str:
        """Вызов LLM с механизмом повторных попыток (Retry)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0, # Максимальная детерминированность
            extra_headers=self._get_headers()
        )
        return response.choices[0].message.content

    def _build_system_prompt(self) -> str:
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        return (
            "Ты — умный агент-исследователь. Твоя задача — отвечать на сложные вопросы, используя инструменты.\n"
            "Используй формат ReAct:\n"
            "THOUGHT: [Твои рассуждения]\n"
            "ACTION: [tool_name]: [input]\n"
            "OBSERVATION: [Результат инструмента]\n"
            "... (повторяй, пока не найдешь ответ)\n"
            "FINAL_ANSWER: [Итоговый ответ]\n\n"
            f"Доступные инструменты:\n{tool_desc}\n\n"
            "ПРАВИЛА:\n"
            "1. Если нужно посчитать физическую величину (время, скорость), ВСЕГДА используй calculator.\n"
            "2. Переводи единицы измерения (км/ч -> м/с) через calculator.\n"
            "3. Не придумывай факты, ищи их в wikipedia."
        )

    def _verify_answer(self, query: str, answer: str) -> str:
        """Шаг самопроверки (Self-Reflection)"""
        logger.info("Запуск самопроверки ответа...")
        verification_msgs = [
            {"role": "system", "content": "Ты — строгий научный рецензент. Проверь ответ на наличие ошибок в логике и физике."},
            {"role": "user", "content": (
                f"Вопрос: {query}\n"
                f"Ответ агента: {answer}\n\n"
                "Проверь:\n"
                "1. Правильно ли выбраны единицы измерения?\n"
                "2. Реалистичен ли ответ?\n"
                "Если всё верно, верни исходный ответ без изменений.\n"
                "Если есть ошибка, исправь её и верни ТОЛЬКО исправленный текст ответа."
            )}
        ]
        return self._llm_call(verification_msgs)

    def solve(self, query: str) -> AgentResponse:
        logger.info(f"Start solving: {query}")
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        trace = []
        final_answer = None

        for step in range(self.max_steps):
            try:
                # 1. Мысль и Действие
                response_text = self._llm_call(messages)
                trace.append(response_text)
                logger.info(f"Step {step+1}: {response_text.splitlines()[0]}...") # Логируем только первую строку мысли

                messages.append({"role": "assistant", "content": response_text})

                # 2. Поиск финального ответа
                if "FINAL_ANSWER:" in response_text:
                    final_answer = response_text.split("FINAL_ANSWER:")[1].strip()
                    break

                # 3. Парсинг действия
                # Ищем паттерн ACTION: tool: query
                match = re.search(r"ACTION:\s*(\w+)\s*:\s*(.*)", response_text, re.IGNORECASE)
                
                if match:
                    tool_name, tool_input = match.group(1).lower(), match.group(2).strip()
                    
                    if tool_name in self.tools:
                        logger.info(f"Calling tool '{tool_name}' with '{tool_input}'")
                        tool_output = self.tools[tool_name].run(tool_input)
                        observation = f"OBSERVATION: {tool_output}"
                    else:
                        observation = f"OBSERVATION: Error: Инструмент {tool_name} не найден."
                    
                    messages.append({"role": "user", "content": observation})
                else:
                    # Если модель не вызвала действие, но и не дала ответ
                    if not final_answer:
                        messages.append({"role": "user", "content": "OBSERVATION: Вы не выполнили ACTION. Пожалуйста, используйте инструмент или дайте FINAL_ANSWER."})

            except Exception as e:
                logger.error(f"Step failed: {e}")
                return AgentResponse(status=AgentStatus.ERROR, error_message=str(e))

        if not final_answer:
            return AgentResponse(status=AgentStatus.ERROR, error_message="Exceeded max iteration steps")

        # 4. Самопроверка
        verified_answer = self._verify_answer(query, final_answer)

        # 5. Формирование ответа по протоколу
        payload = AgentPayload(
            answer=verified_answer,
            reasoning_trace=trace,
            metadata={
                "model": self.model,
                "provider": "openrouter",
                "steps": len(trace)
            }
        )

        return AgentResponse(status=AgentStatus.SUCCESS, payload=payload)

# --- 5. ENTRY POINT ---
if __name__ == "__main__":
    tools_list = [WikipediaTool(), CalculatorTool()]
    agent = OpenRouterAgent(tools=tools_list)

    USER_QUERY = "Сколько понадобится времени гепарду, чтобы пересечь Москву-реку по Большому Каменному мосту?"
    
    # Запуск
    result = agent.solve(USER_QUERY)

    # Вывод (ответ другому сервису)
    print("\n" + "="*40)
    print("A2A JSON OUTPUT:")
    print("="*40)
    print(result.model_dump_json(indent=2, exclude_none=True))