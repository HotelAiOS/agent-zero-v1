#!/usr/bin/env fish
# cleanup-repository-safe.fish - Bezpieczne czyszczenie repozytorium

set REPO_DIR ~/agent-zero-v1
set QUARANTINE_DIR ~/agent-zero-v1-quarantine-(date +%Y%m%d_%H%M)

echo "=== AGENT ZERO V1 - SAFE REPOSITORY CLEANUP ==="
echo "Data: $(date)"
echo ""

# 1. Utwórz katalog kwarantanny
mkdir -p $QUARANTINE_DIR
echo "Quarantine directory: $QUARANTINE_DIR"
echo ""

# 2. Przenieś broken/duplicate files do kwarantanny
echo "2. MOVING BROKEN FILES TO QUARANTINE..."

# Broken docker-compose files
for file in $REPO_DIR/docker-compose.yml.broken-* $REPO_DIR/docker-compose.yml.*healthcheck*
    if test -f $file
        mv $file $QUARANTINE_DIR/
        echo "  Moved: $(basename $file)"
    end
end

# Duplicate agent_zero implementations
for file in $REPO_DIR/agent-zero-*-fixed.py $REPO_DIR/agent-zero-*-working.py $REPO_DIR/agent_zero_phases_*.py
    if test -f $file
        mv $file $QUARANTINE_DIR/
        echo "  Moved: $(basename $file)"
    end
end

# Multiple AI system versions (keep only production_ai_system.py)
for file in $REPO_DIR/*ai_system*.py
    set basename_file (basename $file)
    if test "$basename_file" != "production_ai_system.py"
        mv $file $QUARANTINE_DIR/
        echo "  Moved: $basename_file"
    end
end

# Cleanup scripts (keep only essential)
for file in $REPO_DIR/fix-*.sh $REPO_DIR/deploy-*.sh $REPO_DIR/quick-*.sh
    if test -f $file
        mv $file $QUARANTINE_DIR/
        echo "  Moved: $(basename $file)"
    end
end

# Script duplicates
mv $REPO_DIR/script\ \(*.py $QUARANTINE_DIR/ 2>/dev/null

echo ""

# 3. Przenieś bazy danych do data/
echo "3. ORGANIZING DATABASES..."
mkdir -p $REPO_DIR/data
for file in $REPO_DIR/*.db
    if test -f $file
        mv $file $REPO_DIR/data/
        echo "  Moved to data/: $(basename $file)"
    end
end

echo ""

# 4. Usuń pliki tymczasowe
echo "4. REMOVING TEMPORARY FILES..."
rm -f $REPO_DIR/*.pid
rm -f $REPO_DIR/random
rm -f $REPO_DIR/lsof
rm -f $REPO_DIR/.agent_zero_pids
echo "  Removed PID files and temp files"

echo ""

# 5. Przenieś exported-assets do archive
echo "5. ARCHIVING EXPORTED ASSETS..."
mkdir -p $REPO_DIR/archive
mv $REPO_DIR/exported-assets* $REPO_DIR/archive/ 2>/dev/null
echo "  Moved exported assets to archive/"

echo ""

# 6. Report
echo "=== CLEANUP SUMMARY ==="
echo "Files in quarantine: $(ls $QUARANTINE_DIR | wc -l)"
echo "Quarantine location: $QUARANTINE_DIR"
echo ""
echo "Next steps:"
echo "1. Review quarantined files: ls -lh $QUARANTINE_DIR"
echo "2. If cleanup successful, delete quarantine: rm -rf $QUARANTINE_DIR"
echo "3. Commit cleaned repository: git add -A && git commit -m 'Repository cleanup'"
