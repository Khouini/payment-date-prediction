from datetime import datetime, timedelta
import random
from faker import Faker
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

fake = Faker()

types = ['debit', 'credit']
creditPrices = [400, 500]
debitPrices = [700, 800]
payments = []

# Generate payments for each month of each year from 2019 to 2022
for year in range(2010, 2024):
    for month in range(1, 13):
        num_debit_payments = 15  # Fixed number for all months
        # Generate debit payments
        for _ in range(num_debit_payments):
            day = random.randint(1, 28)  # Simplified: assuming max 28 days to avoid month length issue
            date_time = datetime(year, month, day)
            payment = {
                # "amount": random.randint(700, 800),  # Amount between 500 and 700 for debit payments
                "amount": random.choice(debitPrices),
                "status": "verified",
                "type": "debit",
                "createdAt": date_time.isoformat()
            }
            payments.append(payment)

        # Add credit payments for January and September
        if month in [1, 9]:
            num_credit_payments = 200  # Assuming you want fewer credit payments
            for _ in range(num_credit_payments):
                day = random.randint(1, 28)
                date_time = datetime(year, month, day)
                payment = {
                    # "amount": random.randint(400, 500),  # Amount between 300 and 500 for credit payments
                    "amount": random.choice(creditPrices),
                    "status": "verified",
                    "type": "credit",
                    "createdAt": date_time.isoformat()
                }
                payments.append(payment)

# Log the total number of payments generated
logging.info(f"Generated a total of {len(payments)} payments.")

# calculate the total amount of credit and debit payments
total_credit = sum(payment["amount"] for payment in payments if payment["type"] == "credit")
total_debit = sum(payment["amount"] for payment in payments if payment["type"] == "debit")
print(f"Total credit payments: {total_credit}")
print(f"Total debit payments: {total_debit}")

# calculate the total amount of credit and debit payments for each month
for year in range(2023, 2024):
    for month in range(1, 13):
        total_credit = sum(payment["amount"] for payment in payments if payment["type"] == "credit" and payment["createdAt"].startswith(f"{year}-{month:02d}"))
        total_debit = sum(payment["amount"] for payment in payments if payment["type"] == "debit" and payment["createdAt"].startswith(f"{year}-{month:02d}"))
        print(f"{year}-{month:02d}: Total credit payments: {total_credit}, Total debit payments: {total_debit}")

payments.sort(key=lambda x: datetime.fromisoformat(x["createdAt"]))
# Define the output file path
output_file = "fake_payments.json"

try:
    # Write the payments to the JSON file
    with open(output_file, "w") as f:
        json.dump(payments, f, indent=4)
        logging.info(f"Successfully exported {len(payments)} payments to {output_file}")
except Exception as e:
    logging.error(f"Failed to export payments to {output_file}: {e}")
