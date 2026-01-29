# irctc_analysis.py

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def load_excel(sheet_name):
    return pd.read_excel("Lab Session Data.xlsx", sheet_name=sheet_name)


def mean_manual(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)


def variance_manual(values):
    avg = mean_manual(values)
    total = 0
    for v in values:
        total += (v - avg) ** 2
    return total / len(values)


def average_time(func, values):
    total_time = 0
    for _ in range(10):
        start = time.time()
        func(values)
        total_time += time.time() - start
    return total_time / 10


def main():
    data = load_excel("IRCTC Stock Price")

    prices = data["Price"].values
    change = data["Chg%"].values

    mean_np = np.mean(prices)
    var_np = np.var(prices)

    mean_my = mean_manual(prices)
    var_my = variance_manual(prices)

    mean_np_time = average_time(np.mean, prices)
    mean_my_time = average_time(mean_manual, prices)

    var_np_time = average_time(np.var, prices)
    var_my_time = average_time(variance_manual, prices)

    wednesday_prices = data[data["Day"] == "Wednesday"]["Price"].values
    april_prices = data[data["Month"] == "Apr"]["Price"].values

    wednesday_mean = np.mean(wednesday_prices)
    april_mean = np.mean(april_prices)

    losses = list(filter(lambda x: x < 0, change))
    profits = list(filter(lambda x: x > 0, change))

    loss_prob = len(losses) / len(change)
    profit_prob = len(profits) / len(change)

    wednesday_data = data[data["Day"] == "Wednesday"]
    wednesday_profit = list(filter(lambda x: x > 0, wednesday_data["Chg%"].values))
    wednesday_profit_prob = len(wednesday_profit) / len(wednesday_data)

    print("Population Mean:", mean_np)
    print("Population Variance:", var_np)

    print("\nManual Mean:", mean_my)
    print("Manual Variance:", var_my)

    print("\nAverage Execution Time (10 runs)")
    print("Mean NumPy:", mean_np_time)
    print("Mean Manual:", mean_my_time)
    print("Variance NumPy:", var_np_time)
    print("Variance Manual:", var_my_time)

    print("\nWednesday Mean:", wednesday_mean)
    print("April Mean:", april_mean)

    print("\nProbability of Loss:", loss_prob)
    print("Probability of Profit:", profit_prob)
    print("Probability of Profit on Wednesday:", wednesday_profit_prob)

    plt.scatter(data["Day"], data["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day of Week")
    plt.show()


if __name__ == "__main__":
    main()