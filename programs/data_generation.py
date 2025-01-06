import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

num_samples = 10000

data = pd.DataFrame()

# Generate synthetic data for each essential data element with correlations

# Age -- Normal distribution, range 18-90, correlate 55+ with lower premiums
data['Age'] = np.clip(np.random.normal(40, 15, num_samples), 18, 90).astype(int)
data['Is_Senior'] = (data['Age'] >= 55).astype(int)  # Flag for senior drivers

# Marital Status -- Simulate married vs. non-married
data['Marital_Status'] = np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], num_samples, p=[0.5, 0.3, 0.1, 0.1])
data['Married_Premium_Discount'] = (data['Marital_Status'] == 'Married').astype(int) * 86

# Prior Insurance Coverage -- Categories with longer coverage lowering premiums
data['Prior_Insurance'] = np.random.choice(['<1 year', '1-5 years', '>5 years'], num_samples, p=[0.2, 0.5, 0.3])
data['Prior_Insurance_Premium_Adjustment'] = data['Prior_Insurance'].map({'<1 year': 100, '1-5 years': 50, '>5 years': 0})

# Claims History -- Frequency and severity affect premiums
data['Claims_Frequency'] = np.random.poisson(0.5, num_samples)  # Avg 0.5 claims per year
data['Claims_Severity'] = np.random.choice(['Low', 'Medium', 'High'], num_samples, p=[0.7, 0.2, 0.1])
data['Claims_Adjustment'] = data['Claims_Frequency'] * data['Claims_Severity'].map({'Low': 50, 'Medium': 100, 'High': 200})

# Policy Type -- Categories: liability-only, full coverage
data['Policy_Type'] = np.random.choice(['Liability-Only', 'Full Coverage'], num_samples, p=[0.4, 0.6])
data['Policy_Adjustment'] = data['Policy_Type'].map({'Liability-Only': -200, 'Full Coverage': 0})

# Premium Amount -- Base premium with adjustments
base_premium = 2150
data['Premium_Amount'] = base_premium + data['Married_Premium_Discount'] + data['Prior_Insurance_Premium_Adjustment'] + \
                         data['Claims_Adjustment'] + data['Policy_Adjustment']
data['Premium_Amount'] = np.clip(data['Premium_Amount'], 500, 3000)  # Ensure reasonable limits

# Discounts Applied -- Safe driver, multi-policy, bundling
data['Safe_Driver_Discount'] = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
data['Multi_Policy_Discount'] = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
data['Bundling_Discount'] = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
data['Total_Discounts'] = (data['Safe_Driver_Discount'] + data['Multi_Policy_Discount'] + data['Bundling_Discount']) * 50
data['Premium_Amount'] -= data['Total_Discounts']

# Source of Lead -- Categories with varied conversion probabilities
data['Source_of_Lead'] = np.random.choice(['Online', 'Agent', 'Referral'], num_samples, p=[0.6, 0.3, 0.1])

# Time Since First Contact -- Random days with higher conversion likelihood for shorter times
data['Time_Since_First_Contact'] = np.random.randint(1, 31, num_samples)

# Conversion or Renewal Status -- Based on synthesized data correlations
data['Conversion_Status'] = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])  # 40% conversion rate

# Website Activity -- Correlate engagement with conversion
data['Website_Visits'] = np.random.poisson(5, num_samples)  # Avg 5 visits
data['Conversion_Status'] = np.where(data['Website_Visits'] > 5, 1, data['Conversion_Status'])

# Frequency and Type of Inquiries -- Randomized but weighted by conversion status
data['Inquiries'] = np.random.poisson(2, num_samples)
data['Conversion_Status'] = np.where(data['Inquiries'] > 3, 1, data['Conversion_Status'])

# Number of Quotes Requested -- Simulate higher quotes for undecided customers
data['Quotes_Requested'] = np.random.randint(1, 4, num_samples)

# Time Elapsed Between Initial Quote and Purchase -- Correlate urgency with conversion
data['Time_to_Conversion'] = np.random.randint(1, 15, num_samples)
data['Conversion_Status'] = np.where(data['Time_to_Conversion'] <= 7, 1, data['Conversion_Status'])

# Credit Score or Credit-Based Insurance Score -- Normal distribution with correlation to premiums
data['Credit_Score'] = np.clip(np.random.normal(715, 50, num_samples), 300, 850)
data['Premium_Adjustment_Credit'] = np.where(data['Credit_Score'] > 700, -50, 50)
data['Premium_Amount'] += data['Premium_Adjustment_Credit']

# Zip Code or Region -- Randomly assigned regions with urban vs. rural risk factors
data['Region'] = np.random.choice(['Urban', 'Suburban', 'Rural'], num_samples, p=[0.5, 0.3, 0.2])
data['Premium_Adjustment_Region'] = data['Region'].map({'Urban': 100, 'Suburban': 50, 'Rural': 0})
data['Premium_Amount'] += data['Premium_Adjustment_Region']

# Export synthetic data to CSV
data.to_csv('data/synthetic_insurance_data.csv', index=False)