import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import joblib
import reports

def gui(model, df):

    # Feature order must match training
    features = [
        "BusinessTravel", "JobRole", "MaritalStatus", "OverTime", "Department",
        "AgeGroup", "ExperienceLevel", "IncomeCategory", "LoyaltyCategory",
        "CompanyTenure", "MonthlyIncomeCategory", "IsFarFromHome",
        "IncomePerYearWorked", "FrequentJobChanger", "LoyaltyIndex"
    ]

    # Label encoders for all fields (matching training)
    label_encoders = {
        "BusinessTravel": {"Non-Travel": 0, "Travel_Frequently": 2, "Travel_Rarely": 1},
        "JobRole": {
            "Sales Executive": 7, "Research Scientist": 6, "Laboratory Technician": 3,
            "Manufacturing Director": 4, "Healthcare Representative": 2,
            "Manager": 5, "Sales Representative": 8, "Research Director": 1,
            "Human Resources": 0
        },
        "MaritalStatus": {"Single": 2, "Married": 1, "Divorced": 0},
        "OverTime": {"Yes": 1, "No": 0},
        "Department": {"Sales": 2, "Research & Development": 1, "Human Resources": 0},
        "AgeGroup": {"<30": 0, "30-40": 1, "40-50": 2, "50+": 3},
        "ExperienceLevel": {"Junior": 0, "Mid": 1, "Senior": 2},
        "IncomeCategory": {"Low": 0, "Medium": 1, "High": 2},
        "LoyaltyCategory": {"Low": 0, "Medium": 1, "High": 2},
        "CompanyTenure": {"New": 0, "Moderate": 1, "Long": 2},
        "MonthlyIncomeCategory": {"Low": 0, "Medium": 1, "High": 2},
        "IsFarFromHome": {"No": 0, "Yes": 1},
        "FrequentJobChanger": {"No": 0, "Yes": 1},
        "IncomePerYearWorked": {"Low": 0, "Medium": 1, "High": 2},
        "LoyaltyIndex": {"Low": 0, "Medium": 1, "High": 2}
    }

    # Map categories back to numerical values if the model expects them
    value_maps = {
        "IncomePerYearWorked": {"Low": 4000, "Medium": 7500, "High": 12000},
        "LoyaltyIndex": {"Low": 0.2, "Medium": 0.5, "High": 0.9}
    }

    # Start application
    app = ttk.Window(themename="solar", title="Employee Attrition Predictor")
    app.geometry("950x450")  # increased width for 2 columns

    main_frame = ttk.Frame(app)
    main_frame.pack(fill=BOTH, expand=1, padx=10, pady=10)

    # Scrollable section
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=LEFT, fill=BOTH, expand=1)
    scrollbar.pack(side=RIGHT, fill=Y)

    entries = {}

    # Input generator in 2 columns
    for idx, feature in enumerate(features):
        row = idx // 2
        col = (idx % 2) * 2  # col 0 or 2

        ttk.Label(scrollable_frame, text=feature, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=col, sticky='w', padx=10, pady=6
        )

        if feature in label_encoders:
            cb = ttk.Combobox(scrollable_frame, values=list(label_encoders[feature].keys()), state="readonly", width=35)
            cb.current(0)
            cb.grid(row=row, column=col + 1, pady=6, padx=10)
            entries[feature] = cb
        else:
            e = ttk.Entry(scrollable_frame, width=36)
            e.grid(row=row, column=col + 1, pady=6, padx=10)
            entries[feature] = e

    # Prediction function
    def predict_attrition():
        try:
            input_data = []
            for key in features:
                val = entries[key].get().strip()

                # Use category value maps if needed
                if key in value_maps:
                    input_data.append(value_maps[key][val])
                elif key in label_encoders:
                    input_data.append(label_encoders[key][val])
                else:
                    input_data.append(float(val))

            prediction = model.predict([input_data])[0]
            result = " Likely to Leave" if prediction == 1 else " Not Likely to Leave"
            messagebox.showinfo("Prediction Result", f"Attrition Prediction:\n\n{result}")
        except Exception as e:
            messagebox.showerror("Input Error", f"Check your inputs:\n{e}")

    # Generate Report button function
    def generate_report_button():
        try:
            reports.generate_report()
            messagebox.showinfo("Report", "Report generated successfully!\nCheck 'EDA_Report.pdf'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report:\n{e}")

    # Add Buttons at the bottom center
    button_frame = ttk.Frame(scrollable_frame)
    button_frame.grid(row=(len(features) // 2) + 1, column=0, columnspan=4, pady=30)

    ttk.Button(button_frame, text="Predict Attrition", bootstyle="success", command=predict_attrition).pack(
        side=LEFT, padx=20
    )
    ttk.Button(button_frame, text="EDA Info and Visualization", bootstyle="info", command=generate_report_button).pack(
        side=LEFT, padx=20
    )

    # Run app
    app.mainloop()
