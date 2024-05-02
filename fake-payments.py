from faker import Faker
import random
import json
from datetime import datetime

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

fake = Faker()

methods = ['cheque', 'cash', 'flouci', 'credit_card', 'other']
statuses = ['verified', 'pending', 'failed', 'advanced']
types = ['debit', 'credit']

# Specify the number of payments to generate
num_payments = 1000

payments = []

# Generate the specified number of fake payments
for _ in range(num_payments):
    payment = {
        "amount": random.randint(1, 1000),  # Random amount between 1 and 1000
        "method": random.choice(methods),
        "status": random.choice(statuses),
        "type": random.choice(types),
        "createdAt": fake.date_time_between(start_date='-4y', end_date='-1y').isoformat()  # Random date between 1 year and 4 years ago
    }
    payments.append(payment)

    # Log progress for every 10 payments generated
    if len(payments) % 10 == 0:
        logging.info(f"Generated {len(payments)} payments...")

# Filter payments to include only those within the specified date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 5, 1)

payments.sort(key=lambda x: datetime.fromisoformat(x["createdAt"]))


# filtered_payments = [
#     payment for payment in payments 
#     if start_date <= datetime.fromisoformat(payment["createdAt"]) <= end_date
# ]
filtered_payments = payments

# Log the number of filtered payments
logging.info(f"Total filtered payments within date range: {len(filtered_payments)}")

# Define the output file path
output_file = "fake_payments.json"

try:
    # Write the filtered payments to the JSON file
    with open(output_file, "w") as f:
        json.dump(filtered_payments, f, indent=4)
        logging.info(f"Successfully exported {len(filtered_payments)} payments to {output_file}")
except Exception as e:
    logging.error(f"Failed to export payments to {output_file}: {e}")
