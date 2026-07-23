import sys
import traceback

try:
    import test
    print("Starting test.run_pipeline()...")
    test.run_pipeline()
    print("test.run_pipeline() completed successfully!")
except Exception as e:
    print(f"Error caught during execution: {e}")
    traceback.print_exc()
