# Data Directory

This directory contains datasets and sample files used for model inference and testing.

## Files

- `sample_customers.csv`: Ready-to-use sample CSV containing 50 customer profiles with realistic feature distributions for rapid dashboard testing.

## Feature Dictionary

| Feature Name | Type | Description | Example / Range |
|---|---|---|---|
| `account length` | Integer | Number of months/weeks the customer has stayed with the service | `1` - `243` |
| `area code` | Integer | Telecom area code | `408`, `415`, `510` |
| `international plan` | String | Customer has subscribed to an international calling plan | `'yes'`, `'no'` |
| `voice mail plan` | String | Customer has voicemail service | `'yes'`, `'no'` |
| `number vmail messages` | Integer | Total voicemail messages recorded | `0` - `50` |
| `total day minutes` | Float | Total daytime call duration in minutes | `0.0` - `350.0` |
| `total day calls` | Integer | Total number of daytime calls made | `0` - `160` |
| `total day charge` | Float | Calculated charge for daytime calls | `minutes * ~0.17` |
| `total eve minutes` | Float | Total evening call duration in minutes | `0.0` - `360.0` |
| `total eve calls` | Integer | Total number of evening calls made | `0` - `170` |
| `total eve charge` | Float | Calculated charge for evening calls | `minutes * ~0.085` |
| `total night minutes` | Float | Total night call duration in minutes | `0.0` - `390.0` |
| `total night calls` | Integer | Total number of night calls made | `0` - `170` |
| `total night charge` | Float | Calculated charge for night calls | `minutes * ~0.045` |
| `total intl minutes` | Float | Total international call duration | `0.0` - `20.0` |
| `total intl calls` | Integer | Total international calls made | `0` - `20` |
| `total intl charge` | Float | Calculated charge for international calls | `minutes * ~0.27` |
| `customer service calls`| Integer | Number of calls placed to customer support | `0` - `9` |
