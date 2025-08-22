import os
import random
import math
from typing import List
import numpy as np
import pandas as pd


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


EMPLOYMENT_STATUSES: List[str] = [
    "Salaried",
    "Self-Employed",
    "Contract",
    "Unemployed",
]

LOAN_PURPOSES: List[str] = [
    "Home Renovation",
    "Debt Consolidation",
    "Business Expansion",
    "Education",
    "Car Purchase",
    "Medical",
]


def create_name(idx: int) -> str:
    first_names = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Atharv", "Ishaan",
        "Ananya", "Diya", "Aadhya", "Aarohi", "Anika", "Myra", "Sara", "Aisha",
    ]
    last_names = [
        "Sharma", "Verma", "Iyer", "Kumar", "Patel", "Singh", "Reddy", "Gupta",
        "Das", "Mehta", "Bose", "Nair", "Rao", "Mishra", "Chopra", "Ghosh",
    ]
    return f"{first_names[idx % len(first_names)]} {last_names[(idx // len(first_names)) % len(last_names)]}"


def generate_row(idx: int) -> dict:
    age = int(np.clip(np.random.normal(35, 8), 21, 65))
    employment_status = np.random.choice(EMPLOYMENT_STATUSES, p=[0.6, 0.2, 0.15, 0.05])

    # Base income by employment type
    if employment_status == "Salaried":
        income = int(np.random.lognormal(mean=13.2, sigma=0.35))  # ~ 5L - 20L
    elif employment_status == "Self-Employed":
        income = int(np.random.lognormal(mean=13.4, sigma=0.5))   # slightly higher variance
    elif employment_status == "Contract":
        income = int(np.random.lognormal(mean=12.9, sigma=0.45))
    else:  # Unemployed
        income = int(np.random.lognormal(mean=12.2, sigma=0.5))

    income = int(np.clip(income, 150_000, 4_000_000))

    # Credit score correlated with age and employment stability
    credit_base = 600 + (age - 21) * 2
    credit_noise = np.random.normal(0, 40)
    credit_bump = 30 if employment_status == "Salaried" else (10 if employment_status == "Self-Employed" else -20)
    credit_score = int(np.clip(credit_base + credit_noise + credit_bump, 300, 850))

    # Existing debt as function of income
    existing_debt = int(np.clip(np.random.normal(loc=0.25 * income, scale=0.15 * income), 0, 2_000_000))

    # Loan amount targeted to 6-18x monthly income with noise
    monthly_income = income / 12
    loan_amount = int(np.clip(np.random.normal(12 * monthly_income, 7 * monthly_income), 100_000, 3_000_000))

    loan_purpose = np.random.choice(LOAN_PURPOSES)

    # Simple approval heuristic to create labels with signal
    dti = (existing_debt + 0.8 * loan_amount) / max(income, 1)
    score_component = (credit_score - 650) / 200.0  # centered around 0
    employment_component = {
        "Salaried": 0.25,
        "Self-Employed": 0.1,
        "Contract": -0.05,
        "Unemployed": -0.25,
    }[employment_status]
    risk_raw = 1.2 * dti - score_component - employment_component
    risk_raw += np.random.normal(0, 0.15)

    approved = 1 if risk_raw < 0.9 else 0

    return {
        "applicant_id": 2000 + idx,
        "name": create_name(idx),
        "age": age,
        "income": income,
        "employment_status": employment_status,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "existing_debt": existing_debt,
        "approved": approved,
    }


def generate_dataset(n_rows: int = 2000, out_path: str = "data/applications_large.csv") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = [generate_row(i) for i in range(n_rows)]
    df = pd.DataFrame(rows)
    # Ensure column order matches sample/training schema
    df = df[[
        "applicant_id","name","age","income","employment_status","credit_score",
        "loan_amount","loan_purpose","existing_debt","approved"
    ]]
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = generate_dataset()
    print(f"Wrote {path}")


