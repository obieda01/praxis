#!/bin/bash
cd /gpfs/automountdir/gpfs/homes/SEAS/home/g44758203/praxis

# Stage all updated scripts, results, and configs (excluding large zips/binaries)
git add scripts/ results/ logs/ data/poc_5_tasks.json .gitignore sync.sh

# Commit with timestamp
msg="Sync update: $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$msg"

# Push directly to GitHub main
git push origin main
