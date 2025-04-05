import json
import csv
import ast

def load_test_cases_from_json(test_cases_file):
    with open(test_cases_file, "r") as f:
        return json.load(f)

def load_test_cases_from_csv(test_cases_file):
    with open(test_cases_file, "r") as f:
        reader = csv.DictReader(f)
        test_cases = []
        for row in reader:
            row['nums'] = ast.literal_eval(row['nums'])
            row['expected_nums'] = ast.literal_eval(row['expected_nums'])
            row['expected_k'] = int(row['expected_k'])
            test_cases.append(row)
        return test_cases

def load_test_cases(test_cases_file):
    if test_cases_file.endswith('.json'):
        return load_test_cases_from_json(test_cases_file)
    elif test_cases_file.endswith('.csv'):
        return load_test_cases_from_csv(test_cases_file)
    else:
        raise ValueError("Unsupported file type. Please use a .json or .csv file.")