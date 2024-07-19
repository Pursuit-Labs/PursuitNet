import unittest

class CustomTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_tests = []
        self.failed_tests = []
        self.error_tests = []
        self.skipped_tests = []

    def startTest(self, test):
        super().startTest(test)
        print("========================================================")
        print(f"\n--- Starting Test: {test.id()} ---")

    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"--- Passed: {test} ---\n")
        self.passed_tests.append(test)
        print("========================================================")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"--- Error: {test} ---\n")
        self.error_tests.append(test)
        self.print_error(test, err)
        print("========================================================")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"--- Failed: {test} ---\n")
        self.failed_tests.append(test)
        self.print_error(test, err)
        print("========================================================")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"--- Skipped: {test} ---\nReason: {reason}\n")
        self.skipped_tests.append((test, reason))
        print("========================================================")

    def print_error(self, test, err):
        # Properly format and print the error message
        error_message = self._exc_info_to_string(err, test)
        print(error_message)

    def stopTestRun(self):
        super().stopTestRun()
        print("\n========================================================")
        print("SUMMARY")
        print("========================================================")
        print(f"Passed tests: {len(self.passed_tests)}")
        for test in self.passed_tests:
            print(f"  - {test.id()}")
        print(f"Failed tests: {len(self.failed_tests)}")
        for test in self.failed_tests:
            print(f"  - {test.id()}")
        print(f"Error tests: {len(self.error_tests)}")
        for test in self.error_tests:
            print(f"  - {test.id()}")
        print(f"Skipped tests: {len(self.skipped_tests)}")
        for test, reason in self.skipped_tests:
            print(f"  - {test.id()} (Reason: {reason})")
        print("========================================================")

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
