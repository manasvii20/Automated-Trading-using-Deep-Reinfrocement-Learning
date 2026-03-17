from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def _save_and_plot_results(
    episode_total_assets: Sequence[float],
    drl_lib: str,
    model_name: str,
) -> None:
    """Persist account value trajectory and a simple PnL graph."""
    if not episode_total_assets:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    initial_asset = float(episode_total_assets[0])
    df = pd.DataFrame(
        {
            "step": range(len(episode_total_assets)),
            "total_asset": episode_total_assets,
        }
    )
    df["pnl"] = df["total_asset"] - initial_asset
    df["return"] = df["total_asset"] / initial_asset - 1.0

    base_name = f"account_value_{drl_lib}_{model_name}"
    csv_path = os.path.join(RESULTS_DIR, f"{base_name}.csv")
    png_path = os.path.join(RESULTS_DIR, f"{base_name}.png")

    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["total_asset"], label="Total Asset")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.title(f"Account Value Over Time ({drl_lib}, {model_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def run_buy_and_hold_baseline(
    start_date: str,
    end_date: str,
    ticker_list: list[str],
    data_source: str,
    time_interval: str,
) -> None:
    """Simple equal-weight buy-and-hold baseline over the test window."""
    from finrl.meta.data_processor import DataProcessor

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dp = DataProcessor(data_source)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)

    price_df = (
        data.pivot(index="timestamp", columns="tic", values="close")
        .sort_index()
        .dropna(how="all")
    )
    initial_capital = 1_000_000.0
    first_prices = price_df.iloc[0]
    valid_mask = first_prices > 0
    valid_tickers = first_prices.index[valid_mask]
    first_prices = first_prices[valid_mask]

    if len(valid_tickers) == 0:
        return

    alloc_per_ticker = initial_capital / len(valid_tickers)
    shares = alloc_per_ticker / first_prices

    aligned_prices = price_df[valid_tickers]
    portfolio_values = (aligned_prices * shares).sum(axis=1)

    df = pd.DataFrame(
        {
            "date": portfolio_values.index,
            "account_value": portfolio_values.values,
        }
    )
    df["daily_return"] = df["account_value"].pct_change(1)

    csv_path = os.path.join(RESULTS_DIR, "account_value_baseline_buy_and_hold.csv")
    png_path = os.path.join(RESULTS_DIR, "account_value_baseline_buy_and_hold.png")

    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["account_value"], label="Buy & Hold (Equal Weight)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Buy & Hold Baseline Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor

    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
        "ticker_list": ticker_list,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        _save_and_plot_results(episode_total_assets, drl_lib=drl_lib, model_name=model_name)
        return episode_total_assets
    elif drl_lib == "rllib":
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )
        _save_and_plot_results(episode_total_assets, drl_lib=drl_lib, model_name=model_name)
        return episode_total_assets
    elif drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        _save_and_plot_results(episode_total_assets, drl_lib=drl_lib, model_name=model_name)
        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    # Optional manual demos can be added here if needed.
    pass
