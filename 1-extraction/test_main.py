import pytest
import yaml
import os

from .config_logging import configure_logging
from .test.tests_load import load_test_cases

from .main import Solution

def load_config(config_filename="test/config_test.yaml"):
    """Load configuration from a YAML file."""
    with open(config_filename, "r") as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
test_cases = load_test_cases(config["test_cases_file"])
failure_logger, summary_logger = configure_logging(config)

# Initialize variables to track the number of test cases
total_cases = len(test_cases)
failed_cases = 0

@pytest.mark.parametrize("case", test_cases)
def test_method1(case):
    """Test the functionality of the method being tested.

    This test validates that the method returns the expected output for the given test case.
    If the test fails, the failure details are logged for further analysis.
    """
    global failed_cases
    solution = Solution()
    result = solution.method_to_test(case["input"])  # Replace with the actual method being tested
    
    try:
        # Validate that the result matches the expected output
        assert result == case["expected_result"], f"Expected result: {case['expected_result']}, got: {result}"

    except AssertionError:
        # Log failure details for the test case
        failed_cases += 1
        failure_logger.info(
            f"Test Failure | File: {os.path.basename(config['test_cases_file'])} | "
            f"Test Case ID: {case['id']} | Expected: {case['expected_result']} | "
            f"Got: {result}"
        )
        raise  # Re-raise the exception to preserve the test's failure state


@pytest.fixture(scope="session", autouse=True)
def session_summary_logger():
    """Fixture to log the summary after all tests run.
    
    The code before 'yield' would run before the tests, while the code after 
    'yield' runs after all tests are completed. In this case, there is no setup code before 'yield'.
    """
    yield
    # After all tests, log the total and failed test case counts
    summary_logger.info(f"{os.path.basename(config['test_cases_file'])}|{total_cases}|{failed_cases}")

