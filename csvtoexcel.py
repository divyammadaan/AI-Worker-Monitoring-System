import pandas as pd

# Load CSV
df = pd.read_csv("worker_monitoring_summary.csv")

# Convert Entry and Exit DateTime columns to datetime format
df['Entry_Date_Time'] = pd.to_datetime(df['Entry_Date_Time'])
df['Exit_Date_Time'] = pd.to_datetime(df['Exit_Date_Time'])

# Save original log to Sheet 1
with pd.ExcelWriter("worker_log.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Detailed Log", index=False)

    # Create Summary Sheet: Group by Worker_ID
    summary = df.groupby('Worker_ID').agg(
        Entries=('Entry_Date_Time', 'count'),
        First_Entry=('Entry_Date_Time', 'min'),
        Last_Exit=('Exit_Date_Time', 'max'),
        Total_Duration_Minutes=('Duration', 'sum')
    ).reset_index()

    # Optional: Sort by Total Duration
    summary = summary.sort_values(by="Total_Duration_Minutes", ascending=False)

    # Write summary to Sheet 2
    summary.to_excel(writer, sheet_name="Summary", index=False)

print("Excel file 'worker_log.xlsx' created with Detailed Log and Summary sheets.")
