import json
import csv
from datetime import datetime

# Define input and output file paths
input_file = "fake_payments.json"
output_file = "fake_payments.csv"

try:
    # Read JSON data from input file
    with open(input_file, "r") as f:
        payments = json.load(f)

    # Sort payments by 'createdAt' date before processing
    payments.sort(key=lambda x: datetime.fromisoformat(x["createdAt"]))

    # Extract headers for CSV file (assuming uniform structure across payments)
    headers = list(payments[0].keys())

    # Define headers for CSV columns
    # fieldnames = ["amount", "method", "status", "type", "createdAt"]
    fieldnames = ["amount", "status", "type", "createdAt"]

    # Write payments data to CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers to CSV file
        writer.writeheader()

        # Write each payment as a row in the CSV file
        for payment in payments:
            writer.writerow({
                "amount": payment["amount"],
                # "method": payment["method"],
                "status": payment["status"],
                "type": payment["type"],
                "createdAt": payment["createdAt"]
            })

    print(f"Successfully exported {len(payments)} payments to {output_file}")

except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")

except Exception as e:
    print(f"Error occurred: {e}")
