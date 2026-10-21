#!/bin/bash
cd /gpfs/automountdir/gpfs/homes/SEAS/home/g44758203/praxis

# Auto-refresh VS Code Insiders Git credential helper path if VS Code updated
LATEST_SRV=$(ls -td ~/.vscode-server-insiders/cli/servers/Insiders-*/server 2>/dev/null | head -n 1)
if [ -n "$LATEST_SRV" ] && [ -f "$LATEST_SRV/extensions/git/dist/askpass.sh" ]; then
    export GIT_ASKPASS="$LATEST_SRV/extensions/git/dist/askpass.sh"
    export VSCODE_GIT_ASKPASS_NODE="$LATEST_SRV/node"
    export VSCODE_GIT_ASKPASS_MAIN="$LATEST_SRV/extensions/git/dist/askpass-main.js"
fi

# Stage all updated scripts, results, logs, task JSONs, and Markdown reports
git add scripts/ results/ logs/ data/poc_*.json *.md .gitignore sync.sh

# Commit with timestamp
msg="Sync update: $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$msg"

# Push directly to GitHub main
git push origin main
