import sys
from src.trading.alpaca_client import place_order, should_place_trade,get_current_position
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None



##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    # Add selected analyst nodes in correct order
    prev_node = "start_node"
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge(prev_node, node_name)
        prev_node = node_name  # move pointer

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    """# Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")"""

    # Chain in sequence!
    workflow.add_edge(prev_node, "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow



def parse_args():
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--tickers", type=str, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--show-agent-graph", action="store_true")


    return parser.parse_args()


def select_analysts():
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions:\n1. Space to select\n2. 'a' to select all\n3. Enter to continue\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([
            ("checkbox-selected", "fg:green"),
            ("selected", "fg:green"),
            ("highlighted", "fg:green"),
            ("pointer", "fg:green"),
        ]),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")
    return choices


def select_model():
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    model_info = get_model_info(model_choice)
    provider = model_info.provider.value if model_info else "Unknown"
    print(f"\nSelected {Fore.CYAN}{provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    return model_choice, provider


def build_portfolio(tickers, args):
    return {
        "cash": args.initial_cash,
        "margin_requirement": args.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in tickers
        }
    }


def run_and_execute_trades(tickers, start_date, end_date, portfolio, show_reasoning, selected_analysts, model_choice, model_provider):
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
    )
    print_trading_output(result)

    decisions = result.get("decisions", {})
    if not decisions:
        print("⚠️ No decisions returned from AI.")
        return

    print("\n📥 Executing AI trading decisions...\n")
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        quantity = decision.get("quantity", 0)
        confidence = decision.get("confidence", 0.0)

        action_map = {
            "BUY": "BUY",
            "SELL": "SELL",
            "HOLD": "HOLD",
            "SHORT": "SELL",
        }

        mapped_action = action_map.get(action)
        if mapped_action:
            if should_place_trade(mapped_action, ticker, quantity):
                place_order(mapped_action, ticker, quantity, confidence)
            else:
                print(f"⏭️ Skipping {mapped_action} for {ticker}")
        else:
            print(f"⚠️ Unknown action '{action}' for {ticker}, skipping.")


def initialize_state_with_sentiment(tickers, args, model_choice, model_provider):
    # Step 1: Build the full agent state
    state = AgentState(
        messages=[],
        data={
            "tickers": tickers,
            "portfolio": build_portfolio(tickers, args),
            "start_date": args.start_date or (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d"),
            "end_date": args.end_date or datetime.now().strftime("%Y-%m-%d"),
            "analyst_signals": {},
        },
        metadata={
            "show_reasoning": args.show_reasoning,
            "model_name": model_choice,
            "model_provider": model_provider,
        },
    )

    # Step 2: Run sentiment agent synchronously before anything else
    progress.update_status("system", "all", "Running Sentiment Analysis...")
    sentiment_result = sentiment_agent(state)
    state["data"] = sentiment_result["data"]

    # ✅ Debug print to verify sentiment was generated
    print("✅ [DEBUG] Sentiment scores generated:")
    for ticker, sentiment in state["data"].get("sentiment_scores", {}).items():
        print(f"   {ticker}: score = {sentiment['score']}, summary = {sentiment['summary']}")

    return state

def main():
    args = parse_args()
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # 📌 Step 1: Build empty state with basic fields
    state = initialize_state_with_sentiment(tickers, args, model_choice="gpt-4o", model_provider="OpenAI")

    # 📌 Step 3: Ask user for analyst and model choices
    selected_analysts = select_analysts()
    """if "sentiment_agent" in selected_analysts:
        selected_analysts.remove("sentiment_agent")"""

    model_choice, model_provider = select_model()
    state["metadata"]["model_name"] = model_choice
    state["metadata"]["model_provider"] = model_provider

    # 📌 Step 4: Create and compile the workflow using analyst choices
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        graph_file = "_".join(selected_analysts) + "_graph.png"
        save_graph_as_png(app, graph_file)

    # 📌 Step 5: Run the agents and trades
    run_and_execute_trades(
        tickers,
        state["data"]["start_date"],
        state["data"]["end_date"],
        state["data"]["portfolio"],
        args.show_reasoning,
        selected_analysts,
        model_choice,
        model_provider,
    )



if __name__ == "__main__":
    main()



