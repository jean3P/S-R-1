#!/bin/bash
###############################################################################
# Submit all LeetCode jobs with dependencies
###############################################################################

echo "Submitting LeetCode jobs for 10 problems per difficulty per model..."

# Submit Easy problems first
JOB1=$(sbatch job_leetcode_easy.sbatch | awk '{print $4}')
echo "Submitted Easy job: $JOB1"

# Submit Medium problems after Easy completes
JOB2=$(sbatch job_leetcode_medium.sbatch | awk '{print $4}')
echo "Submitted Medium job: $JOB2 (depends on $JOB1)"

# Submit Hard problems after Medium completes
JOB3=$(sbatch job_leetcode_hard.sbatch | awk '{print $4}')
echo "Submitted Hard job: $JOB3 (depends on $JOB2)"

echo "All jobs submitted successfully!"
echo "Expected completion times:"
echo "  Easy: ~9 hours"
echo "  Medium: ~17.5 hours (cumulative: ~26.5 hours)"
echo "  Hard: ~30 hours (cumulative: ~56.5 hours)"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check job details with: scontrol show job <job_id>"
