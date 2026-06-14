"""
=============================================================================
llm_agent.py  –  Gemini-Powered Defect Analysis Agent
=============================================================================
Provides the DefectAnalysisAgent class that wraps Google Gemini 1.5 Flash
with three callable tools:

  Tool 1 – run_db_query(sql)
            Execute a read-only SELECT over engineering_telemetry.
            Returns a Markdown table of results.

  Tool 2 – trigger_retrain()
            Invoke retrain_pipeline() and return updated model metrics.

  Tool 3 – navigate_dashboard(tab_name, slider_updates)
            Return a state_action dict that the Streamlit app processes
            to switch tabs or adjust sidebar sliders.

Context Injection
-----------------
The system prompt is dynamically built on each instantiation, containing:
  • Project description & dataset overview
  • Database schema (column definitions)
  • Current model evaluation scores (Recall, F1, ROC-AUC)
  • Top-5 highest-risk modules from live DB (if available)
  • Filtered DataFrame head (if supplied by the caller)

Usage
-----
    agent = DefectAnalysisAgent(api_key="...", df=filtered_df, metrics=metrics_dict)
    result = agent.chat("Which modules have complexity > 20?")
    # result = {"text": "...", "tool_calls": [...], "state_action": None | dict}
=============================================================================
"""

import os
import sys
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tab name → index mapping (must match the order in app.py NAV_TABS).
# ─────────────────────────────────────────────────────────────────────────────
TAB_MAP = {
    "overview":         0,
    "executive":        0,
    "summary":          0,
    "deep dive":        1,
    "deep-dive":        1,
    "scatter":          1,
    "metrics":          1,
    "code metrics":     1,
    "ml insights":      2,
    "model":            2,
    "diagnostics":      2,
    "insights":         2,
    "confusion":        2,
    "risk predictor":   3,
    "predictor":        3,
    "simulator":        3,
    "predict":          3,
    "telemetry":        4,
    "api":              4,
    "database":         4,
    "logs":             4,
}


class DefectAnalysisAgent:
    """
    Gemini 1.5 Flash agent with manual function-calling loop.

    Parameters
    ----------
    api_key : str
        Google Generative AI API key.
    df : pd.DataFrame | None
        The currently filtered dataset (for context injection).
    model_metrics : dict | None
        The artefact metrics dict from models/defect_model.pkl.
    """

    def __init__(
        self,
        api_key: str,
        df: Optional[pd.DataFrame] = None,
        model_metrics: Optional[dict] = None,
    ):
        self.df            = df
        self.model_metrics = model_metrics
        self._api_key      = api_key
        self._model        = None
        self._chat         = None
        self.use_groq      = False
        
        # Load Groq key fallback (never hardcoded to comply with push protection)
        try:
            import streamlit as st
            self._groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        except Exception:
            self._groq_api_key = os.environ.get("GROQ_API_KEY", "")

        try:
            self._init_gemini()
        except Exception as e:
            logger.warning("Failed to initialize Gemini agent, falling back to Groq: %s", e)
            self.use_groq = True

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_gemini(self):
        """Initialise the Gemini model with tools and system prompt."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)

            # Register tools as Python callables – Gemini extracts the
            # schema automatically from the docstring + type hints.
            tools = [
                self._tool_run_db_query,
                self._tool_trigger_retrain,
                self._tool_navigate_dashboard,
            ]

            self._model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                tools=tools,
                system_instruction=self._build_system_prompt(),
            )
            # disable_automatic_function_calling=True lets us intercept
            # tool calls ourselves so we can modify Streamlit session state.
            self._chat = self._model.start_chat(
                enable_automatic_function_calling=False
            )
            logger.info("Gemini agent initialised successfully.")

        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as exc:
            logger.error("Failed to initialise Gemini agent: %s", exc)
            raise

    def _build_system_prompt(self) -> str:
        """Construct the rich context-injected system prompt."""
        # ── Project Description ────────────────────────────────────────────
        base = """You are CodeSentinel AI, an expert software quality intelligence assistant
embedded in the NASA Metrics Data Program (MDP) Defect Prediction Dashboard.

You have direct access to a live SQLite database of software module telemetry,
a trained Random Forest defect prediction model, and the currently displayed
dataset in the Streamlit dashboard.

Your capabilities:
1. Answer analytical questions about code quality, defect risk, and engineering trends.
2. Execute SQL queries against the live engineering_telemetry table.
3. Trigger model retraining when asked.
4. Navigate the dashboard to a specific tab on the user's behalf. Note that you are running inside a persistent floating chat panel in the bottom-right corner, so there is no separate tab for you. Valid tabs are: overview (Tab 0), deep-dive (Tab 1), ml insights (Tab 2), risk predictor (Tab 3), and telemetry (Tab 4).

Always be concise, data-driven, and actionable. If you run a SQL query, summarise
the results in plain language after showing the table.
"""

        # ── Database Schema ────────────────────────────────────────────────
        schema = """
DATABASE SCHEMA  (SQLite table: engineering_telemetry)
------------------------------------------------------
id                  INTEGER  – Auto-increment primary key
source              TEXT     – 'historical' | 'api' | 'ci_hook' | 'manual'
predicted_risk      REAL     – Model defect probability at logging time
logged_at           TEXT     – UTC ISO-8601 timestamp
loc_blank           REAL     – Blank lines of code
branch_count        REAL     – Branch count (McCabe input)
loc_code_and_comment REAL    – Lines containing both code and comments
loc_comments        REAL     – Comment lines
cyclomatic_complexity REAL   – McCabe v(g) – PRIMARY complexity metric
design_complexity   REAL     – iv(g) design complexity
essential_complexity REAL    – ev(g) essential complexity
loc_executable      REAL     – Executable lines of code
halstead_content    REAL     – Halstead vocabulary size
halstead_difficulty REAL     – Halstead difficulty D
halstead_effort     REAL     – Halstead effort E
halstead_error_est  REAL     – Estimated number of bugs (B = E^(2/3) / 3000)
halstead_length     REAL     – Halstead program length N
halstead_level      REAL     – Halstead level L
halstead_prog_time  REAL     – Estimated programming time (seconds)
halstead_volume     REAL     – Halstead volume V
num_operands        REAL     – Total operands N2
num_operators       REAL     – Total operators N1
num_unique_operands REAL     – Unique operands η2
num_unique_operators REAL    – Unique operators η1
loc_total           REAL     – Total lines of code
defective           INTEGER  – Ground truth label (0=clean, 1=defective)
"""

        # ── Model Metrics ──────────────────────────────────────────────────
        metrics_section = "\nCURRENT MODEL PERFORMANCE\n--------------------------\n"
        if self.model_metrics:
            m = self.model_metrics
            metrics_section += (
                f"Algorithm      : Random Forest (300 trees, SMOTE-balanced)\n"
                f"Recall         : {m.get('recall', 'N/A'):.4f}  ← Primary metric\n"
                f"F1-Score       : {m.get('f1', 'N/A'):.4f}\n"
                f"Precision      : {m.get('precision', 'N/A'):.4f}\n"
                f"ROC-AUC        : {m.get('roc_auc', 'N/A'):.4f}\n"
                f"Decision Threshold: {m.get('threshold', 'N/A'):.2f}\n"
                f"Top features   : {', '.join(m.get('feature_names', [])[:5])}\n"
            )
        else:
            metrics_section += "Model not yet loaded.\n"

        # ── Live DB Summary ────────────────────────────────────────────────
        db_section = "\nLIVE DATABASE SNAPSHOT\n----------------------\n"
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.database import get_stats, execute_query
            stats = get_stats()
            db_section += (
                f"Total records  : {stats.get('total_rows', 'N/A'):,}\n"
                f"Defective      : {stats.get('defective_count', 'N/A'):,} "
                f"({stats.get('defect_rate_pct', 'N/A')}%)\n"
                f"Last logged at : {stats.get('last_logged_at', 'N/A')}\n"
            )
            top5 = execute_query(
                "SELECT id, cyclomatic_complexity, halstead_volume, loc_total, "
                "defective, predicted_risk "
                "FROM engineering_telemetry "
                "ORDER BY predicted_risk DESC NULLS LAST "
                "LIMIT 5"
            )
            if not top5.empty:
                db_section += f"\nTop-5 highest predicted risk modules:\n{top5.to_markdown(index=False)}\n"
        except Exception as exc:
            db_section += f"(DB summary unavailable: {exc})\n"

        # ── DataFrame Head ─────────────────────────────────────────────────
        df_section = "\nCURRENT FILTERED DATAFRAME (first 8 rows)\n------------------------------------------\n"
        if self.df is not None and not self.df.empty:
            df_section += self.df.head(8).to_markdown(index=False)
        else:
            df_section += "(No DataFrame currently available.)\n"

        return base + schema + metrics_section + db_section + df_section

    # ──────────────────────────────────────────────────────────────────────
    # Tool Definitions (exposed to Gemini)
    # ──────────────────────────────────────────────────────────────────────

    def _tool_run_db_query(self, sql: str) -> str:
        """
        Execute a read-only SQL SELECT query against the engineering_telemetry
        SQLite database and return the results as a Markdown table.

        Args:
            sql: A valid SQLite SELECT statement.

        Returns:
            A Markdown-formatted table of query results, or an error message.
        """
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from src.database import execute_query
            df = execute_query(sql)
            if df.empty:
                return "Query returned 0 rows."
            return df.to_markdown(index=False)
        except Exception as exc:
            return f"Query error: {exc}"

    def _tool_trigger_retrain(self) -> str:
        """
        Trigger a full model retraining cycle using all records in the live
        SQLite database.  Applies SMOTE and RandomizedSearchCV, then overwrites
        the production model artefact (models/defect_model.pkl).

        Returns:
            A summary string of the new model evaluation metrics after retraining.
        """
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from src.train_model import retrain_pipeline
            metrics = retrain_pipeline()
            return (
                f"✅ Retraining complete!\n"
                f"  Recall    : {metrics['recall']:.4f}\n"
                f"  F1-Score  : {metrics['f1']:.4f}\n"
                f"  Precision : {metrics['precision']:.4f}\n"
                f"  ROC-AUC   : {metrics['roc_auc']:.4f}\n"
                f"  Threshold : {metrics['threshold']:.2f}\n"
                "The production model has been updated. Please refresh the dashboard."
            )
        except Exception as exc:
            return f"Retraining failed: {exc}"

    def _tool_navigate_dashboard(self, tab_name: str, highlight_metric: str = "") -> str:
        """
        Navigate the Streamlit dashboard to a specific tab.

        Args:
            tab_name: Name of the target tab.  Supported values:
                      overview, deep-dive, ml insights, risk predictor,
                      telemetry (case-insensitive).
            highlight_metric: Optional metric or feature name to highlight
                              on the target tab.

        Returns:
            Confirmation string.  The caller processes the navigation action.
        """
        key = tab_name.lower().strip()
        if key in ["ai assistant", "chat", "assistant"]:
            return "I am already open in the floating panel in the bottom-right corner! You can chat with me here directly."
        tab_index = TAB_MAP.get(key)
        if tab_index is None:
            # Fuzzy fallback: check if any key is contained in the input
            for k, v in TAB_MAP.items():
                if k in key or key in k:
                    tab_index = v
                    break
        if tab_index is None:
            return (
                f"Unknown tab: '{tab_name}'. "
                f"Valid options: {list(set(TAB_MAP.values()))}"
            )

        # Store as a pending action that app.py will pick up after this call.
        self._pending_state_action = {
            "type":    "navigate",
            "tab":     tab_index,
            "highlight": highlight_metric,
        }
        return f"Navigating to tab {tab_index} ({tab_name})."

    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    # Groq Fallback Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _chat_groq(self, message: str) -> dict:
        """Fallback chat using Groq Cloud REST API with Llama-3.3-70b."""
        import httpx
        import json

        if not hasattr(self, "_groq_history"):
            self._groq_history = []

        self._groq_history.append({"role": "user", "content": message})

        # Describe the tools in standard JSON schema format for Groq/OpenAI API
        groq_tools = [
            {
                "type": "function",
                "function": {
                    "name": "_tool_run_db_query",
                    "description": "Execute a read-only SQL SELECT query against the engineering_telemetry SQLite database and return the results as a Markdown table.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "A valid SQLite SELECT statement."
                            }
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "_tool_trigger_retrain",
                    "description": "Trigger a full model retraining cycle using all records in the live SQLite database. Applies SMOTE and RandomizedSearchCV, then overwrites the production model artefact.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "_tool_navigate_dashboard",
                    "description": "Navigate the Streamlit dashboard to a specific tab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tab_name": {
                                "type": "string",
                                "description": "Name of the target tab: overview, deep-dive, ml insights, risk predictor, telemetry."
                            },
                            "highlight_metric": {
                                "type": "string",
                                "description": "Optional metric or feature name to highlight on the target tab."
                            }
                        },
                        "required": ["tab_name"]
                    }
                }
            }
        ]

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._groq_api_key}",
            "Content-Type": "application/json"
        }

        tool_calls_made = []
        self._pending_state_action = None

        # Function calling loop
        for _ in range(5):
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": self._build_system_prompt()}] + self._groq_history,
                "tools": groq_tools,
                "tool_choice": "auto"
            }
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
                if response.status_code != 200:
                    logger.error("Groq API error: %s", response.text)
                    return {
                        "text": f"⚠️ Groq API Error (Status {response.status_code}): {response.text}",
                        "tool_calls": tool_calls_made,
                        "state_action": None
                    }
                res_data = response.json()
                choice = res_data["choices"][0]
                res_msg = choice["message"]

                # Check for tool calls
                if "tool_calls" in res_msg and res_msg["tool_calls"]:
                    self._groq_history.append(res_msg)

                    for tc in res_msg["tool_calls"]:
                        fn_name = tc["function"]["name"]
                        fn_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        tc_id = tc["id"]

                        logger.info("Groq tool call: %s(%s)", fn_name, fn_args)
                        tool_calls_made.append(fn_name)

                        # Dispatch
                        tool_map = {
                            "_tool_run_db_query":        self._tool_run_db_query,
                            "_tool_trigger_retrain":     self._tool_trigger_retrain,
                            "_tool_navigate_dashboard":  self._tool_navigate_dashboard,
                        }
                        fn = tool_map.get(fn_name)
                        if fn is None:
                            result_str = f"Unknown tool: {fn_name}"
                        else:
                            try:
                                result_str = fn(**fn_args) if fn_args else fn()
                            except Exception as exc:
                                result_str = f"Tool execution error: {exc}"

                        self._groq_history.append({
                            "role": "tool",
                            "name": fn_name,
                            "tool_call_id": tc_id,
                            "content": result_str
                        })
                    continue
                else:
                    self._groq_history.append(res_msg)
                    return {
                        "text": res_msg.get("content", ""),
                        "tool_calls": tool_calls_made,
                        "state_action": self._pending_state_action
                    }
            except Exception as e:
                logger.error("Error in Groq fallback chat: %s", e)
                return {
                    "text": f"⚠️ Groq fallback chat error: {e}",
                    "tool_calls": tool_calls_made,
                    "state_action": None
                }
        return {
            "text": "I completed the requested action.",
            "tool_calls": tool_calls_made,
            "state_action": self._pending_state_action
        }

    def chat(self, message: str) -> dict:
        """
        Send a user message, execute any requested tool calls, and return
        the final assistant reply (falls back transparently to Groq if Gemini fails).
        """
        if self.use_groq:
            return self._chat_groq(message)

        import google.generativeai as genai

        tool_calls_made = []
        self._pending_state_action = None   # Reset before each turn.

        try:
            response = self._chat.send_message(message)

            # ── Function-calling loop (max 5 iterations) ──────────────────
            for _ in range(5):
                # Check if the response contains a function call.
                fc_parts = [
                    p for p in response.candidates[0].content.parts
                    if hasattr(p, "function_call") and p.function_call.name
                ]
                if not fc_parts:
                    break   # No more function calls – we have the final answer.

                function_responses = []
                for part in fc_parts:
                    fc   = part.function_call
                    name = fc.name
                    args = dict(fc.args)

                    logger.info("Tool call: %s(%s)", name, args)
                    tool_calls_made.append(name)

                    # Dispatch to the right tool method.
                    tool_map = {
                        "_tool_run_db_query":        self._tool_run_db_query,
                        "_tool_trigger_retrain":     self._tool_trigger_retrain,
                        "_tool_navigate_dashboard":  self._tool_navigate_dashboard,
                    }
                    fn = tool_map.get(name)
                    if fn is None:
                        result_str = f"Unknown tool: {name}"
                    else:
                        try:
                            result_str = fn(**args) if args else fn()
                        except Exception as exc:
                            result_str = f"Tool execution error: {exc}"

                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=name,
                                response={"result": result_str},
                            )
                        )
                    )

                # Send all function results back to the model in one turn.
                response = self._chat.send_message(
                    genai.protos.Content(
                        parts=function_responses,
                        role="user",
                    )
                )

            # ── Extract final text ────────────────────────────────────────
            text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

            return {
                "text":         text or "I completed the requested action.",
                "tool_calls":   tool_calls_made,
                "state_action": self._pending_state_action,
            }

        except Exception as exc:
            logger.warning("Gemini chat failed, falling back dynamically to Groq: %s", exc)
            self.use_groq = True
            return self._chat_groq(message)
