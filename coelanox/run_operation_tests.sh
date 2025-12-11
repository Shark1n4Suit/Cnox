#!/bin/bash
# Script to run comprehensive operation tests and generate a summary report

cd /home/shark1n4uit/Saic/coelanox/coelanox-packager

echo "Running comprehensive operation tests..."
echo "This may take a few minutes and may encounter segfaults..."
echo ""

# Run the test and capture output
cargo test --test test_all_operations_comprehensive test_all_operations_comprehensive -- --nocapture 2>&1 | tee /tmp/test_output.txt

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

# Extract summary from output
grep -A 20 "TEST SUMMARY" /tmp/test_output.txt || echo "Summary not found in output"

# Extract operations that passed
echo ""
echo "Operations that PASSED both 1D and 2D tests:"
grep -A 50 "OPERATIONS PASSING BOTH TESTS" /tmp/test_output.txt | grep "✅" || echo "None found"

# Extract operations that failed or segfaulted
echo ""
echo "Operations that FAILED or SEGFAULTED:"
grep -A 50 "FAILED OPERATIONS" /tmp/test_output.txt || echo "None found"

# Count operations
echo ""
echo "Statistics:"
grep -E "(Passed:|Failed:|Segfaults:|Skipped:)" /tmp/test_output.txt || echo "Statistics not found"

