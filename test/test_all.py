import unittest

class CustomTestResult(unittest.TestResult):
    def startTest(self, test):
        super().startTest(test)
        print("========================================================")
        print(f"\n--- Starting Test: {test.id()} ---")

    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"--- Passed: {test} ---\n")
        print("========================================================")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"--- Error: {test} ---\n")
        self.print_error(test, err)
        print("========================================================")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"--- Failed: {test} ---\n")
        self.print_error(test, err)
        print("========================================================")

    def print_error(self, test, err):
        # Properly format and print the error message
        error_message = self._exc_info_to_string(err, test)
        print(error_message)

class CustomTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _makeResult(self):
        return CustomTestResult()

def suite():
    # Load all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="test", pattern="test_*.py")
    return test_suite

if __name__ == "__main__":
    runner = CustomTestRunner(verbosity=2)
    runner.run(suite())
